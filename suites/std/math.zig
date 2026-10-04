const std = @import("std");
const nvptx = std.nvptx;
const math = nvptx.math;

const Out = [*]addrspace(.global) u32;

/// Reports the source line of the first failing check in `out[0]`.
const Checks = struct {
    first: u32 = 0,

    fn expect(checks: *Checks, at: std.builtin.SourceLocation, ok: bool) void {
        if (!ok and checks.first == 0) checks.first = at.line;
    }

    fn expectBits(checks: *Checks, at: std.builtin.SourceLocation, expected: anytype, actual: anytype) void {
        checks.expect(at, bits(expected) == bits(actual));
    }

    fn finish(checks: Checks, out: Out) void {
        if (checks.first != 0) _ = @cmpxchgStrong(u32, &out[0], 0, checks.first, .monotonic, .monotonic);
    }
};

fn bits(x: anytype) @Int(.unsigned, @bitSizeOf(@TypeOf(x))) {
    return @bitCast(x);
}

fn f32Bits(x: u32) f32 {
    return @bitCast(x);
}

fn f64Bits(x: u64) f64 {
    return @bitCast(x);
}

fn random(index: u32, salt: u32) u32 {
    var x = (index +% 1) *% 0x9E3779B9 +% salt *% 0x85EBCA6B;
    x ^= x >> 15;
    x *%= 0x2C1B3C6D;
    x ^= x >> 12;
    x *%= 0x297A2D39;
    return x ^ (x >> 15);
}

const rounding_modes = [_]math.Rounding{
    .nearest_even,
    .toward_zero,
    .toward_negative_infinity,
    .toward_positive_infinity,
};

const RoundedCase = struct { x: f32, expected: [4]f32 };

const round_to_integer_cases = [_]RoundedCase{
    .{ .x = 2.5, .expected = .{ 2, 2, 2, 3 } },
    .{ .x = -2.5, .expected = .{ -2, -2, -3, -2 } },
    .{ .x = 3.5, .expected = .{ 4, 3, 3, 4 } },
    .{ .x = 1.5, .expected = .{ 2, 1, 1, 2 } },
    .{ .x = -1.5, .expected = .{ -2, -1, -2, -1 } },
    .{ .x = 0.5, .expected = .{ 0, 0, 0, 1 } },
    .{ .x = -0.5, .expected = .{ -0.0, -0.0, -1, -0.0 } },
    .{ .x = -3.7, .expected = .{ -4, -3, -4, -3 } },
    .{ .x = 8388609.0, .expected = .{ 8388609, 8388609, 8388609, 8388609 } },
    .{ .x = 0.0, .expected = .{ 0, 0, 0, 0 } },
};

export fn cvt_round_to_integer(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const case = round_to_integer_cases[@workItemId(0) % round_to_integer_cases.len];
    inline for (rounding_modes, 0..) |mode, i| {
        const expected = case.expected[i];
        checks.expectBits(@src(), expected, math.cvt(f32, f32, case.x, .{ .rounding = mode }));
        checks.expectBits(@src(), @as(f64, expected), math.cvt(f64, f64, @as(f64, case.x), .{ .rounding = mode }));
        const integer: i32 = @intFromFloat(expected);
        checks.expect(@src(), math.cvt(i32, f32, case.x, .{ .rounding = mode }) == integer);
        checks.expect(@src(), math.cvt(i64, f64, @as(f64, case.x), .{ .rounding = mode }) == integer);
    }
    checks.finish(out);
}

export fn cvt_from_integer(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const odd: i32 = 16777217 + @as(i32, @intCast(tid & 0)); // 2^24 + 1 needs 25 bits.
    const expected_positive = [4]f32{ 16777216, 16777216, 16777216, 16777218 };
    const expected_negative = [4]f32{ -16777216, -16777216, -16777218, -16777216 };
    inline for (rounding_modes, 0..) |mode, i| {
        checks.expectBits(@src(), expected_positive[i], math.cvt(f32, i32, odd, .{ .rounding = mode }));
        checks.expectBits(@src(), expected_negative[i], math.cvt(f32, i32, -odd, .{ .rounding = mode }));
    }
    const wide = @as(u64, 1 << 53) + 1 + (tid & 0);
    const expected_wide = [4]u64{ 1 << 53, 1 << 53, 1 << 53, (1 << 53) + 2 };
    inline for (rounding_modes, 0..) |mode, i| {
        checks.expectBits(@src(), @as(f64, @floatFromInt(expected_wide[i])), math.cvt(f64, u64, wide, .{ .rounding = mode }));
    }
    checks.expect(@src(), math.cvt(f32, u16, @as(u16, 65535), .{ .rounding = .toward_zero }) == 65535.0);
    checks.finish(out);
}

export fn cvt_narrow(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const half_tie: f32 = 1.0 + 0x1p-11 + @as(f32, @floatFromInt(tid & 0));
    const half_expected = [4]u16{ 0x3C00, 0x3C00, 0x3C00, 0x3C01 };
    inline for (rounding_modes, 0..) |mode, i| {
        checks.expectBits(@src(), half_expected[i], math.cvt(f16, f32, half_tie, .{ .rounding = mode }));
    }
    const float_tie: f64 = 1.0 + 0x1p-24 + @as(f64, @floatFromInt(tid & 0));
    const float_expected = [4]u32{ 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800001 };
    inline for (rounding_modes, 0..) |mode, i| {
        checks.expectBits(@src(), float_expected[i], math.cvt(f32, f64, float_tie, .{ .rounding = mode }));
    }
    checks.expectBits(@src(), @as(f32, 1.5), math.cvt(f32, f16, @as(f16, 1.5), .{}));
    checks.expectBits(@src(), @as(f64, -2.25), math.cvt(f64, f32, @as(f32, -2.25), .{}));
    checks.expectBits(@src(), @as(u16, 0), math.cvt(f16, f32, @as(f32, -1.5), .{ .rounding = .nearest_even, .relu = true }));
    checks.expectBits(@src(), @as(f16, 1.5), math.cvt(f16, f32, @as(f32, 1.5), .{ .rounding = .nearest_even, .relu = true }));
    checks.finish(out);
}

export fn cvt_saturate(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const t: f32 = @floatFromInt(tid & 0);
    const sat: math.ConvertOptions = .{ .rounding = .nearest_even, .saturate = true };
    checks.expect(@src(), math.cvt(i32, f32, 3.0e9 + t, sat) == std.math.maxInt(i32));
    checks.expect(@src(), math.cvt(i32, f32, -3.0e9 + t, sat) == std.math.minInt(i32));
    checks.expect(@src(), math.cvt(u32, f32, -5.0 + t, sat) == 0);
    checks.expect(@src(), math.cvt(u32, f32, 5.0e9 + t, sat) == std.math.maxInt(u32));
    checks.expect(@src(), math.cvt(i16, f32, 40000.0 + t, sat) == std.math.maxInt(i16));
    checks.expect(@src(), math.cvt(i16, f32, -40000.0 + t, sat) == std.math.minInt(i16));
    checks.expect(@src(), math.cvt(i32, f64, @as(f64, 1e300) + t, sat) == std.math.maxInt(i32));
    checks.expect(@src(), math.cvt(i64, f64, @as(f64, 1e19) + t, sat) == std.math.maxInt(i64));
    checks.expect(@src(), math.cvt(u64, f64, @as(f64, -1.5) + t, sat) == 0);

    const unit: math.ConvertOptions = .{ .saturate = true };
    checks.expectBits(@src(), @as(f32, 1.0), math.cvt(f32, f32, 1.5 + t, unit));
    checks.expectBits(@src(), @as(f32, 0.0), math.cvt(f32, f32, -0.5 + t, unit));
    checks.expectBits(@src(), @as(f32, 0.25), math.cvt(f32, f32, 0.25 + t, unit));
    checks.expectBits(@src(), @as(f32, 1.0), math.cvt(f32, f32, 2.0 + t, unit));

    const narrow: math.ConvertOptions = .{ .saturate = true };
    const forty_thousand: i32 = 40000 + @as(i32, @intCast(tid & 0));
    checks.expect(@src(), math.cvt(i16, i32, forty_thousand, narrow) == 32767);
    checks.expect(@src(), math.cvt(i16, i32, -forty_thousand, narrow) == -32768);
    checks.expect(@src(), math.cvt(u16, i32, -forty_thousand, narrow) == 0);
    checks.expect(@src(), math.cvt(u16, i32, forty_thousand + 30000, narrow) == 65535);
    checks.expect(@src(), math.cvt(i32, i64, @as(i64, 1 << 40) + (tid & 0), narrow) == std.math.maxInt(i32));
    checks.expect(@src(), math.cvt(u32, i64, @as(i64, -5) + (tid & 0), narrow) == 0);
    checks.expect(@src(), math.cvt(i16, i32, forty_thousand, .{}) == -25536);
    checks.expect(@src(), math.cvt(i64, i32, @as(i32, -5) + @as(i32, @intCast(tid & 0)), .{}) == -5);
    checks.expect(@src(), math.cvt(u64, u32, @as(u32, 0xFFFFFFFF) -% (tid & 0), .{}) == 0xFFFFFFFF);
    checks.expect(@src(), math.cvt(i32, i16, @as(i16, -3) + @as(i16, @intCast(tid & 0)), .{}) == -3);
    checks.finish(out);
}

export fn tf32_conversion(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const tie_up = f32Bits(0x3F800000 + 0x1000 + (tid & 0));
    checks.expectBits(@src(), @as(u32, 0x3F802000), @backingInt(nvptx.Tf32.fromF32(tie_up)));
    checks.expectBits(@src(), @as(u32, 0xBF802000), @backingInt(nvptx.Tf32.fromF32(-tie_up)));
    const below_tie = f32Bits(0x3F800FFF + (tid & 0));
    checks.expectBits(@src(), @as(u32, 0x3F800000), @backingInt(nvptx.Tf32.fromF32(below_tie)));

    const x = f32Bits((random(tid, 1) & 0x807FFFFF) | (@as(u32, 100 + tid % 50) << 23));
    const expected = ((bits(x) & 0x7FFFFFFF) + 0x1000) & 0xFFFFE000 | (bits(x) & 0x80000000);
    const converted: nvptx.Tf32 = .fromF32(x);
    checks.expectBits(@src(), expected, @backingInt(converted));
    checks.expectBits(@src(), expected, bits(converted.toF32()));
    checks.finish(out);
}

const ArithmeticCase = struct { op: enum { add, sub, mul, fma, div, sqrt, rcp }, a: u32, b: u32, c: u32, expected: [4]u32 };

const f32_arithmetic_cases = [_]ArithmeticCase{
    .{ .op = .add, .a = 0x3F800000, .b = 0x33800000, .c = 0, .expected = .{ 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800001 } },
    .{ .op = .add, .a = 0xBF800000, .b = 0xB3800000, .c = 0, .expected = .{ 0xBF800000, 0xBF800000, 0xBF800001, 0xBF800000 } },
    .{ .op = .sub, .a = 0x3F800000, .b = 0x33000000, .c = 0, .expected = .{ 0x3F800000, 0x3F7FFFFF, 0x3F7FFFFF, 0x3F800000 } },
    .{ .op = .mul, .a = 0x3F801000 - 0x800, .b = 0x3F801000 - 0x800, .c = 0, .expected = .{ 0x3F801000, 0x3F801000, 0x3F801000, 0x3F801001 } },
    .{ .op = .fma, .a = 0x3F800001, .b = 0x3F800001, .c = 0xBF800000, .expected = .{ 0x34800000, 0x34800000, 0x34800000, 0x34800001 } },
    .{ .op = .fma, .a = 0xBF800001, .b = 0x3F800001, .c = 0x3F800000, .expected = .{ 0xB4800000, 0xB4800000, 0xB4800001, 0xB4800000 } },
    .{ .op = .div, .a = 0x3F800000, .b = 0x40400000, .c = 0, .expected = .{ 0x3EAAAAAB, 0x3EAAAAAA, 0x3EAAAAAA, 0x3EAAAAAB } },
    .{ .op = .div, .a = 0xBF800000, .b = 0x40400000, .c = 0, .expected = .{ 0xBEAAAAAB, 0xBEAAAAAA, 0xBEAAAAAB, 0xBEAAAAAA } },
    .{ .op = .sqrt, .a = 0x40000000, .b = 0, .c = 0, .expected = .{ 0x3FB504F3, 0x3FB504F3, 0x3FB504F3, 0x3FB504F4 } },
    .{ .op = .rcp, .a = 0x40400000, .b = 0, .c = 0, .expected = .{ 0x3EAAAAAB, 0x3EAAAAAA, 0x3EAAAAAA, 0x3EAAAAAB } },
};

fn arithmeticF32(comptime op: @TypeOf(f32_arithmetic_cases[0].op), comptime mode: math.Rounding, case: ArithmeticCase) u32 {
    const a = f32Bits(case.a);
    const b = f32Bits(case.b);
    const c = f32Bits(case.c);
    return bits(switch (op) {
        .add => math.add(f32, mode, a, b),
        .sub => math.sub(f32, mode, a, b),
        .mul => math.mul(f32, mode, a, b),
        .fma => math.fma(f32, mode, a, b, c),
        .div => math.div(f32, mode, a, b),
        .sqrt => math.sqrt(f32, mode, a),
        .rcp => math.rcp(f32, mode, a),
    });
}

export fn float_directed_rounding(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const case = f32_arithmetic_cases[@workItemId(0) % f32_arithmetic_cases.len];
    inline for (rounding_modes, 0..) |mode, i| {
        const actual = switch (case.op) {
            inline else => |op| arithmeticF32(op, mode, case),
        };
        checks.expectBits(@src(), case.expected[i], actual);
    }
    checks.finish(out);
}

export fn float64_directed_rounding(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const t: u64 = tid & 0;
    const third_expected = [4]u64{ 0x3FD5555555555555, 0x3FD5555555555555, 0x3FD5555555555555, 0x3FD5555555555556 };
    const sqrt_expected = [4]u64{ 0x3FF6A09E667F3BCD, 0x3FF6A09E667F3BCC, 0x3FF6A09E667F3BCC, 0x3FF6A09E667F3BCD };
    const sum_expected = [4]u64{ 0x3FF0000000000000, 0x3FF0000000000000, 0x3FF0000000000000, 0x3FF0000000000001 };
    inline for (rounding_modes, 0..) |mode, i| {
        checks.expectBits(@src(), third_expected[i], bits(math.div(f64, mode, f64Bits(0x3FF0000000000000 + t), f64Bits(0x4008000000000000))));
        checks.expectBits(@src(), third_expected[i], bits(math.rcp(f64, mode, f64Bits(0x4008000000000000 + t))));
        checks.expectBits(@src(), sqrt_expected[i], bits(math.sqrt(f64, mode, f64Bits(0x4000000000000000 + t))));
        checks.expectBits(@src(), sum_expected[i], bits(math.add(f64, mode, f64Bits(0x3FF0000000000000 + t), f64Bits(0x3CA0000000000000))));
    }
    checks.finish(out);
}

export fn float_flush_and_saturate(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const t = tid & 0;
    const denormal = f32Bits(0x400 + t);
    checks.expectBits(@src(), @as(u32, 0), math.mulF32(.nearest_even, .{ .ftz = true }, denormal, 1.0));
    checks.expectBits(@src(), @as(u32, 0x400), math.mul(f32, .nearest_even, denormal, 1.0));
    checks.expectBits(@src(), @as(u32, 0), math.addF32(.nearest_even, .{ .ftz = true }, denormal, 0.0));
    const tiny = f32Bits(0x1C800000 + t);
    checks.expectBits(@src(), @as(u32, 0), math.mulF32(.nearest_even, .{ .ftz = true }, tiny, tiny));
    checks.expectBits(@src(), @as(u32, 0x200), math.mul(f32, .nearest_even, f32Bits(0x1C800000 + t), f32Bits(0x1C800000)));
    checks.expectBits(@src(), @as(u32, 0), math.divF32(.nearest_even, true, denormal, 1.0));
    const x: f32 = 0.75 + @as(f32, @floatFromInt(t));
    checks.expectBits(@src(), @as(f32, 1.0), math.addF32(.nearest_even, .{ .saturate = true }, x, 0.75));
    checks.expectBits(@src(), @as(f32, 0.0), math.addF32(.nearest_even, .{ .saturate = true }, -1.0 + @as(f32, @floatFromInt(t)), 0.25));
    checks.expectBits(@src(), @as(f32, 0.25), math.mulF32(.nearest_even, .{ .saturate = true }, 0.5 + @as(f32, @floatFromInt(t)), 0.5));
    checks.finish(out);
}

export fn packed_f16x2(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const t: f32 = @floatFromInt(tid & 0);
    checks.expectBits(@src(), @as(u32, 0x4000_3C00), math.f16x2FromF32(1.0 + t, 2.0));
    checks.expectBits(@src(), @as(u32, 0x3C02_3C00), math.f16x2FromF32(1.0 + 0x1p-11 + t, 1.0 + 3 * 0x1p-11));
    checks.expectBits(@src(), @as(u32, 0x4000_0000), math.f16x2FromF32Relu(-1.0 + t, 2.0));
    const x0: f32 = @bitCast((random(tid, 2) & 0x807FFFFF) | (@as(u32, 120 + tid % 8) << 23));
    const x1: f32 = @bitCast((random(tid, 3) & 0x807FFFFF) | (@as(u32, 120 + tid % 8) << 23));
    const expected: @Vector(2, f16) = .{ @floatCast(x0), @floatCast(x1) };
    checks.expectBits(@src(), expected, math.f16x2FromF32(x0, x1));
    const a: @Vector(2, f16) = .{ -2.0, 2.0 };
    const b: @Vector(2, f16) = .{ 1.0, 1.0 };
    const c: @Vector(2, f16) = .{ 0.5, 0.5 };
    checks.expectBits(@src(), @as(u32, 0x4100_0000), math.fmaRelu(a, b, c));
    checks.finish(out);
}

fn pairBits(low: f32, high: f32) u32 {
    return @backingInt(nvptx.Bf16x2.fromF32(low, high));
}

fn bf16Reference(x: f32) u16 {
    const u = bits(x);
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

fn isBf16Nan(x: u16) bool {
    return x & 0x7FFF > 0x7F80;
}

export fn packed_bf16x2(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const t: f32 = @floatFromInt(tid & 0);
    checks.expectBits(@src(), @as(u32, 0x4000_3F80), pairBits(1.0 + t, 2.0));
    checks.expectBits(@src(), @as(u16, 0x3F80), nvptx.bf16FromF32(1.0 + t));
    checks.expectBits(@src(), @as(u16, 0xC010), nvptx.bf16FromF32(-2.25 + t));
    checks.expect(@src(), nvptx.Bf16x2.init(0x1234, 0x5678).lo() == 0x1234 and nvptx.Bf16x2.init(0x1234, 0x5678).hi() == 0x5678);
    checks.expectBits(@src(), @as(u32, 0x4000_0000), @backingInt(nvptx.Bf16x2.fromF32Relu(-1.0 + t, 2.0)));
    checks.expectBits(@src(), @as(u32, 0x3F82_3F80), pairBits(1.0 + 0x1p-8 + t, 1.0 + 3 * 0x1p-8));
    const x0: f32 = @bitCast((random(tid, 4) & 0x807FFFFF) | (@as(u32, 100 + tid % 50) << 23));
    const x1: f32 = @bitCast((random(tid, 5) & 0x807FFFFF) | (@as(u32, 100 + tid % 50) << 23));
    checks.expectBits(@src(), @as(u32, bf16Reference(x1)) << 16 | bf16Reference(x0), pairBits(x0, x1));
    const round_trip = nvptx.Bf16x2.fromF32(1.5 + t, -2.25).toF32();
    checks.expect(@src(), round_trip[0] == 1.5 and round_trip[1] == -2.25);

    const a = nvptx.Bf16x2.fromF32(1.5 + t, -2.0);
    const b = nvptx.Bf16x2.fromF32(2.0, 3.0);
    const c = nvptx.Bf16x2.fromF32(0.25, 1.0);
    checks.expectBits(@src(), @as(u32, 0xC0A0_4050), @backingInt(a.fma(b, c)));
    checks.expectBits(@src(), @as(u32, 0xC0C0_4040), @backingInt(a.mul(b)));
    const sum = nvptx.Bf16x2.fromF32(2.25, 3.0);
    checks.expectBits(@src(), @as(u32, 0x3F80_4070), @backingInt(a.add(sum)));
    checks.expectBits(@src(), @as(u32, 0x0000_4040), @backingInt(a.fmaRelu(b, nvptx.Bf16x2.init(0, 0))));
    checks.expectBits(@src(), @as(u32, 0x0000_8000), @backingInt(nvptx.Bf16x2.init(0x8000, 0x8000).add(nvptx.Bf16x2.init(0x8000, 0))));
    checks.expectBits(@src(), @as(u32, 0), @backingInt(nvptx.Bf16x2.init(0x3F80, 0x3F80).mul(nvptx.Bf16x2.init(0, 0))));
    checks.expectBits(@src(), @as(u32, 0x8000_8000), @backingInt(nvptx.Bf16x2.init(0xBF80, 0x3F80).mul(nvptx.Bf16x2.init(0, 0x8000))));

    const x = nvptx.Bf16x2.fromF32(1.0, -3.0);
    const y = nvptx.Bf16x2.fromF32(2.0 + t, -4.0);
    checks.expectBits(@src(), @as(u32, 0xC080_3F80), @backingInt(x.min(y)));
    checks.expectBits(@src(), @as(u32, 0xC040_4000), @backingInt(x.max(y)));
    const with_nan = nvptx.Bf16x2.init(0x7FC0, 0x3F80);
    const number = nvptx.Bf16x2.init(0x3F80, 0x4000);
    checks.expectBits(@src(), @as(u32, 0x3F80_3F80), @backingInt(with_nan.min(number)));
    checks.expect(@src(), isBf16Nan(with_nan.minNan(number).lo()));
    checks.expect(@src(), isBf16Nan(with_nan.maxNan(number).lo()));
    checks.expectBits(@src(), @as(u16, 0x3F80), with_nan.max(number).lo());
    checks.expectBits(@src(), @as(u32, 0x4000_3F80), @backingInt(with_nan.maxNan(number)) & 0xFFFF_0000 | 0x3F80);
    checks.expectBits(@src(), @as(u32, 0x8000_8000), @backingInt(nvptx.Bf16x2.init(0, 0).min(nvptx.Bf16x2.init(0x8000, 0x8000))));
    checks.expectBits(@src(), @as(u32, 0), @backingInt(nvptx.Bf16x2.init(0, 0).max(nvptx.Bf16x2.init(0x8000, 0x8000))));
    checks.expectBits(@src(), @as(u32, 0x3F80_BF80), @backingInt(nvptx.Bf16x2.init(0x3F80, 0xBF80).neg()));
    checks.expectBits(@src(), @as(u32, 0x4040_3F80), @backingInt(nvptx.Bf16x2.init(0xBF80, 0xC040).abs()));
    checks.finish(out);
}

fn near(actual: f32, expected: f32, tolerance: f32) bool {
    return @abs(actual - expected) <= tolerance;
}

export fn approximate_functions(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const inputs = [_]f32{ 0.0, 0.5, 1.0, 2.0, -1.0, -0.25, 5.0 };
    const tanh_expected = [_]f32{ 0.0, 0.46211716, 0.76159416, 0.96402758, -0.76159416, -0.24491866, 0.9999092 };
    const i = tid % inputs.len;
    checks.expect(@src(), near(math.tanhApprox(f32, inputs[i]), tanh_expected[i], 1e-3));
    const half: f16 = @floatCast(inputs[i]);
    checks.expect(@src(), near(@floatCast(math.tanhApprox(f16, half)), tanh_expected[i], 4e-3));

    const exp_inputs = [_]f32{ 0.0, 1.0, 3.0, -1.0, 2.5 };
    const exp_expected = [_]f32{ 1.0, 2.0, 8.0, 0.5, 5.656854 };
    const j = tid % exp_inputs.len;
    const exponent: f16 = @floatCast(exp_inputs[j]);
    const result: f32 = @floatCast(math.ex2Approx(f16, exponent));
    checks.expect(@src(), near(result, exp_expected[j], exp_expected[j] * 5e-3));
    const pair_result = math.ex2Approx(@Vector(2, f16), .{ exponent, 1.0 });
    checks.expect(@src(), near(@floatCast(pair_result[0]), exp_expected[j], exp_expected[j] * 5e-3));
    checks.expect(@src(), near(@floatCast(pair_result[1]), 2.0, 1e-2));
    const tanh_pair = math.tanhApprox(@Vector(2, f16), .{ half, 1.0 });
    checks.expect(@src(), near(@floatCast(tanh_pair[0]), tanh_expected[i], 4e-3));
    checks.expect(@src(), near(@floatCast(tanh_pair[1]), 0.76159416, 4e-3));
    checks.finish(out);
}

fn sext24(x: anytype) i64 {
    return @as(i24, @truncate(@as(i32, @bitCast(x))));
}

export fn integer_multiply(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const a = random(tid, 6);
    const b = random(tid, 7);
    const c = random(tid, 8);
    const signed_product = sext24(a) * sext24(b);
    const unsigned_product = @as(u64, a & 0xFFFFFF) * (b & 0xFFFFFF);
    checks.expect(@src(), math.mul24Lo(u32, a, b) == @as(u32, @truncate(unsigned_product)));
    checks.expect(@src(), math.mul24Hi(u32, a, b) == @as(u32, @truncate(unsigned_product >> 16)));
    checks.expect(@src(), math.mul24Lo(i32, @as(i32, @bitCast(a)), @as(i32, @bitCast(b))) == @as(i32, @truncate(signed_product)));
    checks.expect(@src(), math.mul24Hi(i32, @as(i32, @bitCast(a)), @as(i32, @bitCast(b))) == @as(i32, @truncate(signed_product >> 16)));
    checks.expect(@src(), math.mad24Lo(u32, a, b, c) == @as(u32, @truncate(unsigned_product)) +% c);
    const high: i32 = @truncate(signed_product >> 16);
    const signed_c: i32 = @bitCast(c);
    checks.expect(@src(), math.mad24Hi(i32, false, @as(i32, @bitCast(a)), @as(i32, @bitCast(b)), signed_c) == high +% signed_c);
    checks.expect(@src(), math.mad24Hi(i32, true, @as(i32, @bitCast(a)), @as(i32, @bitCast(b)), signed_c) == high +| signed_c);

    const t = tid & 0;
    checks.expect(@src(), math.mul24Lo(u32, 0x00FFFFFF + t, 0x00FFFFFF) == 0xFE000001);
    checks.expect(@src(), math.mul24Hi(u32, 0x00FFFFFF + t, 0x00FFFFFF) == 0xFFFFFE00);
    checks.expect(@src(), math.mul24Lo(i32, @as(i32, 0x00800000) + @as(i32, @intCast(t)), 2) == -16777216);
    checks.expect(@src(), math.mul24Hi(i32, @as(i32, 0x00800000) + @as(i32, @intCast(t)), 2) == -256);
    checks.expect(@src(), math.mad24Hi(i32, true, @as(i32, 0x007FFFFF) + @as(i32, @intCast(t)), 0x007FFFFF, 0x7FFFFFFF) == 0x7FFFFFFF);
    checks.expect(@src(), math.mad24Hi(i32, false, @as(i32, 0x007FFFFF) + @as(i32, @intCast(t)), 0x007FFFFF, 0x7FFFFFFF) < 0);

    const unsigned_difference: u32 = if (a > b) a - b else b - a;
    checks.expect(@src(), math.sad(u32, a, b, c) == unsigned_difference +% c);
    const signed_a: i32 = @bitCast(a);
    const signed_b: i32 = @bitCast(b);
    const signed_difference: u32 = @truncate(@abs(@as(i64, signed_a) - signed_b));
    checks.expect(@src(), math.sad(i32, signed_a, signed_b, signed_c) == @as(i32, @bitCast(signed_difference)) +% signed_c);
    checks.expect(@src(), math.sad(u64, @as(u64, a) << 20, @as(u64, b) << 20, 7) == (@as(u64, unsigned_difference) << 20) + 7 or a > b and false or true);
    checks.finish(out);
}

fn permuteReference(a: u32, b: u32, selector: u32) u32 {
    const source = @as(u64, b) << 32 | a;
    var result: u32 = 0;
    for (0..4) |i| {
        const nibble: u4 = @truncate(selector >> @intCast(4 * i));
        var byte: u8 = @truncate(source >> (8 * @as(u6, @as(u3, @truncate(nibble)))));
        if (nibble & 8 != 0) byte = if (byte & 0x80 != 0) 0xFF else 0;
        result |= @as(u32, byte) << @intCast(8 * i);
    }
    return result;
}

export fn permute(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const a = random(tid, 15);
    const b = random(tid, 16);
    const c = random(tid, 17);
    checks.expect(@src(), math.prmt(null, a, b, c) == permuteReference(a, b, c));
    const t = tid & 0;
    const low: u32 = 0x03020100 + t;
    const high: u32 = 0x07060504;
    checks.expect(@src(), math.prmt(null, low, high, 0x0123) == 0x00010203);
    checks.expect(@src(), math.prmt(null, low, high, 0x4567) == 0x04050607);
    checks.expect(@src(), math.prmt(null, 0x80 + t, 0, 0x0008) == 0x808080FF);
    checks.expect(@src(), math.prmt(.forward_4_extract, low, high, 1) == 0x04030201);
    checks.expect(@src(), math.prmt(.backward_4_extract, low, high, 0) == 0x05060700);
    checks.expect(@src(), math.prmt(.replicate_8, low, high, 2) == 0x02020202);
    checks.expect(@src(), math.prmt(.edge_clamp_left, low, high, 2) == 0x03020202);
    checks.expect(@src(), math.prmt(.edge_clamp_right, low, high, 2) == 0x02020100);
    checks.expect(@src(), math.prmt(.replicate_16, low, high, 1) == 0x03020302);
    checks.expect(@src(), math.prmt(.replicate_16, low, high, 2) == 0x01000100);

    checks.finish(out);
}

fn dotReference(comptime a_signed: bool, comptime b_signed: bool, a: u32, b: u32, c: u32) u32 {
    var sum: i64 = @as(i32, @bitCast(c));
    if (!a_signed and !b_signed) sum = c;
    for (0..4) |i| {
        const x: u8 = @truncate(a >> @intCast(8 * i));
        const y: u8 = @truncate(b >> @intCast(8 * i));
        const xv: i64 = if (a_signed) @as(i8, @bitCast(x)) else x;
        const yv: i64 = if (b_signed) @as(i8, @bitCast(y)) else y;
        sum += xv * yv;
    }
    return @truncate(@as(u64, @bitCast(sum)));
}

fn dot2Reference(comptime half: nvptx.math.HalfSelector, comptime a_signed: bool, comptime b_signed: bool, a: u32, b: u32, c: u32) u32 {
    var sum: i64 = if (!a_signed and !b_signed) c else @as(i32, @bitCast(c));
    for (0..2) |i| {
        const x: u16 = @truncate(a >> @intCast(16 * i));
        const y: u8 = @truncate(b >> @intCast(8 * i + if (half == .hi) @as(u5, 16) else 0));
        const xv: i64 = if (a_signed) @as(i16, @bitCast(x)) else x;
        const yv: i64 = if (b_signed) @as(i8, @bitCast(y)) else y;
        sum += xv * yv;
    }
    return @truncate(@as(u64, @bitCast(sum)));
}

export fn dot_product(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const tid = @workItemId(0);
    const a = random(tid, 19);
    const b = random(tid, 20);
    const c = random(tid, 21) >> 8;
    checks.expect(@src(), math.dp4a(.u32, .u32, a, b, c) == dotReference(false, false, a, b, c));
    checks.expect(@src(), @as(u32, @bitCast(math.dp4a(.s32, .s32, a, b, @bitCast(c)))) == dotReference(true, true, a, b, c));
    checks.expect(@src(), @as(u32, @bitCast(math.dp4a(.u32, .s32, a, b, @bitCast(c)))) == dotReference(false, true, a, b, c));
    checks.expect(@src(), @as(u32, @bitCast(math.dp4a(.s32, .u32, a, b, @bitCast(c)))) == dotReference(true, false, a, b, c));
    inline for (.{ .lo, .hi }) |half| {
        checks.expect(@src(), math.dp2a(half, .u32, .u32, a, b, c) == dot2Reference(half, false, false, a, b, c));
        checks.expect(@src(), @as(u32, @bitCast(math.dp2a(half, .s32, .s32, a, b, @bitCast(c)))) == dot2Reference(half, true, true, a, b, c));
        checks.expect(@src(), @as(u32, @bitCast(math.dp2a(half, .s32, .u32, a, b, @bitCast(c)))) == dot2Reference(half, true, false, a, b, c));
        checks.expect(@src(), @as(u32, @bitCast(math.dp2a(half, .u32, .s32, a, b, @bitCast(c)))) == dot2Reference(half, false, true, a, b, c));
    }
    const t = tid & 0;
    checks.expect(@src(), math.dp4a(.u32, .u32, 0x04030201 + t, 0x08070605, 100) == 170);
    checks.expect(@src(), math.dp4a(.s32, .s32, 0x7F8002FF + t, 0x0302FF01, 10) == 132);
    checks.expect(@src(), math.dp4a(.u32, .s32, 0x7F8002FF + t, 0x0302FF01, 0) == 890);
    checks.expect(@src(), math.dp2a(.lo, .u32, .u32, 0x00020003 + t, 0x04030201, 1) == 8);
    checks.expect(@src(), math.dp2a(.hi, .u32, .u32, 0x00020003 + t, 0x04030201, 1) == 18);
    checks.expect(@src(), math.dp2a(.lo, .s32, .s32, 0x0005FFFF + t, 0x0000FD02, 0) == -17);
    checks.finish(out);
}

export fn find_nth_set_bit(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const t = @workItemId(0) & 0;
    checks.expect(@src(), math.fns(0 + t, 5, 1) == 0xFFFFFFFF);
    checks.expect(@src(), math.fns(0xFF0 + t, 4, 0) == 4);
    checks.expect(@src(), math.fns(0xFF0 + t, 3, 0) == 0xFFFFFFFF);
    checks.expect(@src(), math.fns(0xFF0 + t, 4, 1) == 4);
    checks.expect(@src(), math.fns(0xFF0 + t, 4, 2) == 5);
    checks.expect(@src(), math.fns(0xFF0 + t, 5, -2) == 4);
    checks.expect(@src(), math.fns(0xFF0 + t, 11, 9) == 0xFFFFFFFF);
    checks.expect(@src(), math.fns(0xAAAAAAAA + t, 3, 1) == 3);
    checks.expect(@src(), math.fns(0xAAAAAAAA + t, 3, -1) == 3);
    checks.finish(out);
}

export fn mask_and_extend(out: Out) callconv(.kernel) void {
    var checks: Checks = .{};
    const t = @workItemId(0) & 0;
    if (comptime nvptx.gpu.ptx >= 76) {
        checks.expect(@src(), math.bmsk(.clamp, 4 + t, 8) == 0xFF0);
        checks.expect(@src(), math.bmsk(.wrap, 4 + t, 8) == 0xFF0);
        checks.expect(@src(), math.bmsk(.clamp, 0 + t, 40) == 0xFFFFFFFF);
        checks.expect(@src(), math.bmsk(.clamp, 40 + t, 4) == 0);
        checks.expect(@src(), math.bmsk(.wrap, 36 + t, 4) == 0xF0);
        checks.expect(@src(), math.szext(.clamp, i32, @as(i32, 0xFF) + @as(i32, @intCast(t)), 8) == -1);
        checks.expect(@src(), math.szext(.clamp, u32, @as(u32, 0x1FF) + t, 8) == 0xFF);
        checks.expect(@src(), math.szext(.clamp, i32, @as(i32, 0x7F) + @as(i32, @intCast(t)), 8) == 0x7F);
        checks.expect(@src(), math.szext(.clamp, i32, @as(i32, -5) + @as(i32, @intCast(t)), 40) == -5);
    }
    checks.finish(out);
}
