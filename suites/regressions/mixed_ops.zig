//! Each function reads and writes one local variable through accesses of different sizes: union
//! members, `@bitCast` of aggregates and pointer casts. The functions run unchanged on the host
//! and on the GPU, and the build checks that ptxas gives the kernels of `mixed_kernels.zig` no
//! stack frame and no spilled registers.

const samples = [_]u64{
    0,          1,                  0x7fffffff,         0x80000000,
    0xffffffff, 0x123456789abcdef0, 0xfedcba9876543210, 0xffffffffffffffff,
    0x7f80ff7f, 0x00ff00ff80ff807f, 0x7fc01234ffc0abcd, 0x7ff8000000000001,
};

/// The host and the launcher fill `out` with 0xdeadbeef, so the result is `sample` and the
/// compiler cannot know it.
const seed_index = 1023;

inline fn operand(out: anytype, sample: u64) u64 {
    return sample ^ (out[seed_index] ^ 0xdeadbeef);
}

inline fn low(x: u64) u32 {
    return @truncate(x);
}

inline fn high(x: u64) u32 {
    return @truncate(x >> 32);
}

const Wide = extern union { whole: u64, words: [2]u32, shorts: [4]u16, bytes: [8]u8 };

pub fn wide_halves(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var wide: Wide = .{ .whole = x };
        out[n] = wide.words[0];
        out[n + 1] = wide.words[1];
        wide.words[1] = ~low(x);
        out[n + 2] = low(wide.whole);
        out[n + 3] = high(wide.whole);
        wide.words[0] = high(x) ^ 0xa5a5a5a5;
        out[n + 4] = high(wide.whole);
        out[n + 5] = low(wide.whole);
        n += 6;
    }
}

pub fn narrow_stores(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var wide: Wide = .{ .whole = x };
        wide.bytes[1] = @truncate(x >> 3);
        wide.shorts[3] = @truncate(x >> 9);
        wide.bytes[4] = @truncate(x >> 17);
        wide.words[0] ^= 0x01010101;
        out[n] = low(wide.whole);
        out[n + 1] = high(wide.whole);
        n += 2;
    }
}

pub fn sign_extension(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        const wide: Wide = .{ .whole = x };
        const signed_bytes: [8]i8 = @bitCast(wide.bytes);
        const signed_shorts: [4]i16 = @bitCast(wide.shorts);
        var bytes_sum: u32 = 0;
        var shorts_sum: u32 = 0;
        inline for (0..8) |i| {
            const signed: i32 = signed_bytes[i];
            const unsigned: u32 = wide.bytes[i];
            bytes_sum = bytes_sum *% 31 +% @as(u32, @bitCast(signed)) +% unsigned *% 7;
        }
        inline for (0..4) |i| {
            const signed: i32 = signed_shorts[i];
            const unsigned: u32 = wide.shorts[i];
            shorts_sum = shorts_sum *% 31 +% @as(u32, @bitCast(signed)) +% unsigned *% 7;
        }
        out[n] = bytes_sum;
        out[n + 1] = shorts_sum;
        out[n + 2] = @bitCast(@as(i32, signed_bytes[3]));
        out[n + 3] = @bitCast(@as(i32, signed_shorts[2]));
        out[n + 4] = @bitCast(@as(i32, @as(i8, @bitCast(wide.bytes[7]))));
        n += 5;
    }
}

const Pair = extern union { whole: u32, shorts: [2]u16, bytes: [4]u8, single: f32 };

pub fn word_bytes(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = low(operand(out, sample));
        var pair: Pair = .{ .whole = x };
        pair.bytes[2] +%= 3;
        const shorts_before = pair.shorts[1];
        pair.shorts[0] = @truncate(x >> 5);
        out[n] = pair.whole;
        out[n + 1] = shorts_before;
        out[n + 2] = pair.bytes[0];
        out[n + 3] = @as(u32, @bitCast(@as(i32, @as(i8, @bitCast(pair.bytes[3])))));
        n += 4;
    }
}

pub fn float_payloads(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var pair: Pair = .{ .whole = low(x) };
        const single = pair.single;
        pair.shorts[1] = @truncate(x >> 40);
        out[n] = @bitCast(single);
        out[n + 1] = @bitCast(pair.single);
        var halves: [2]f16 = @bitCast(low(x));
        const first = halves[0];
        halves[0] = halves[1];
        halves[1] = first;
        out[n + 2] = @as(u32, @bitCast(halves));
        const doubles: [2]f32 = @bitCast(x);
        out[n + 3] = @bitCast(doubles[1]);
        const merged: f64 = @bitCast([2]f32{ doubles[1], doubles[0] });
        const merged_bits: u64 = @bitCast(merged);
        out[n + 4] = low(merged_bits);
        out[n + 5] = high(merged_bits);
        n += 6;
    }
}

pub fn double_halves(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        const value: f64 = @bitCast(x);
        const parts: [2]u32 = @bitCast(value);
        out[n] = parts[0];
        out[n + 1] = parts[1];
        const swapped: f64 = @bitCast([2]u32{ parts[1], parts[0] });
        const swapped_bits: u64 = @bitCast(swapped);
        out[n + 2] = low(swapped_bits);
        out[n + 3] = high(swapped_bits);
        n += 4;
    }
}

pub fn pointer_views(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var whole: u64 = x;
        const words: *[2]u32 = @ptrCast(&whole);
        words[1] +%= 0x10001;
        const bytes: *[8]u8 = @ptrCast(&whole);
        bytes[0] = @truncate(x >> 11);
        bytes[5] ^= 0xff;
        out[n] = low(whole);
        out[n + 1] = high(whole);
        const shorts: *[4]u16 = @ptrCast(&whole);
        out[n + 2] = shorts[1];
        out[n + 3] = shorts[2];
        n += 4;
    }
}

const Quad = struct { a: u8, b: u8, c: u16 };

pub fn packed_fields(out: anytype) void {
    const Fields = packed struct(u32) { a: u8, b: u16, c: u8 };
    var n: usize = 0;
    for (samples) |sample| {
        const x = low(operand(out, sample));
        var fields: Fields = @bitCast(x);
        fields.b +%= 0x101;
        fields.c = ~fields.c;
        const back: u32 = @bitCast(fields);
        const quad: Quad = .{ .a = fields.a, .b = fields.c, .c = fields.b };
        const as_bytes: [4]u8 = .{ quad.a, quad.b, @truncate(quad.c), @truncate(quad.c >> 8) };
        out[n] = back;
        out[n + 1] = @bitCast(as_bytes);
        n += 2;
    }
}

const Choice = union(enum) { word: u32, half: u16, byte: u8 };

pub fn tagged_union(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var choice: Choice = switch (@as(u2, @truncate(x))) {
            0 => .{ .word = low(x) },
            1 => .{ .half = @truncate(x >> 8) },
            else => .{ .byte = @truncate(x >> 16) },
        };
        switch (choice) {
            .word => |*word| word.* +%= 1,
            .half => |*half| half.* ^= 0xf0f0,
            .byte => |*byte| byte.* = ~byte.*,
        }
        out[n] = switch (choice) {
            .word => |word| word,
            .half => |half| @as(u32, half) << 8,
            .byte => |byte| @as(u32, byte) << 16,
        };
        out[n + 1] = switch (choice) {
            .word => 1,
            .half => 2,
            .byte => 3,
        };
        n += 2;
    }
}

pub fn conditional_stores(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var wide: Wide = .{ .whole = x };
        if (x & 1 != 0) wide.bytes[1] = @truncate(x >> 24);
        if (x & 2 != 0) wide.shorts[2] = @truncate(x >> 7);
        if (x & 4 != 0) wide.words[0] = high(x);
        out[n] = low(wide.whole);
        out[n + 1] = high(wide.whole);
        var count: u32 = 0;
        while (count < (x >> 8) % 4) : (count += 1) {
            if (count % 2 == 0) wide.bytes[1] +%= 5 else wide.bytes[5] +%= 5;
            wide.shorts[0] +%= @truncate(count);
        }
        out[n + 2] = low(wide.whole);
        out[n + 3] = high(wide.whole);
        n += 4;
    }
}

pub fn unwritten_bytes(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x = operand(out, sample);
        var wide: Wide = undefined;
        wide.words[1] = high(x);
        wide.bytes[0] = @truncate(x);
        const defined_high = wide.shorts[3];
        wide.shorts[1] = @truncate(x >> 3);
        out[n] = wide.words[1];
        out[n + 1] = defined_high;
        out[n + 2] = wide.bytes[0];
        out[n + 3] = wide.shorts[1];
        n += 4;
    }
}

pub const names = [_][]const u8{
    "wide_halves",
    "narrow_stores",
    "sign_extension",
    "word_bytes",
    "float_payloads",
    "double_halves",
    "pointer_views",
    "packed_fields",
    "tagged_union",
    "conditional_stores",
    "unwritten_bytes",
};
