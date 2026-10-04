const std = @import("std");
const nv = std.nvptx;
const matrix = nv.matrix;

const Element = matrix.Element;
const Accumulator = matrix.Accumulator;

const region_words = 512;

var shared_words: [4 * region_words]u32 align(32) addrspace(.shared) = undefined;

fn elementWidth(comptime element: Element) u32 {
    return switch (element) {
        .f16, .bf16 => 16,
        .tf32 => 32,
        .e4m3, .e5m2, .s8, .u8 => 8,
        .s4, .u4 => 4,
        .b1 => 1,
        .f64 => 64,
    };
}

fn isUnsigned(comptime element: Element) bool {
    return element == .u8 or element == .u4;
}

fn floatBits(comptime T: type, v: i32) u64 {
    const Bits = @Int(.unsigned, @bitSizeOf(T));
    return @as(Bits, @bitCast(@as(T, @floatFromInt(v))));
}

fn encodeElement(comptime element: Element, v: i32) u64 {
    const bits: u32 = @bitCast(v);
    return switch (element) {
        .f16 => floatBits(f16, v),
        .bf16 => floatBits(f32, v) >> 16,
        .tf32 => floatBits(f32, v),
        .f64 => floatBits(f64, v),
        .e4m3 => e4m3Bits(v),
        .e5m2 => floatBits(f16, v) >> 8,
        .s8, .u8 => bits & 0xff,
        .s4, .u4 => bits & 0xf,
        .b1 => bits & 1,
    };
}

/// Returns the e4m3 encoding (1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits) of a small
/// integer with at most 4 significant bits.
fn e4m3Bits(v: i32) u64 {
    if (v == 0) return 0;
    const bits: u32 = @truncate(floatBits(f32, v));
    const exponent = (bits >> 23 & 0xff) - 127 + 7;
    return (bits >> 31) << 7 | exponent << 3 | (bits >> 20 & 7);
}

fn encodeAccumulator(comptime accumulator: Accumulator, v: i32) u64 {
    return switch (accumulator) {
        .f16 => floatBits(f16, v),
        .f32 => floatBits(f32, v),
        .s32 => @as(u32, @bitCast(v)),
        .f64 => floatBits(f64, v),
    };
}

fn decodeAccumulator(comptime accumulator: Accumulator, bits: u64) f64 {
    return switch (accumulator) {
        .f16 => @floatCast(@as(f16, @bitCast(@as(u16, @truncate(bits))))),
        .f32 => @floatCast(@as(f32, @bitCast(@as(u32, @truncate(bits))))),
        .s32 => @floatFromInt(@as(i32, @bitCast(@as(u32, @truncate(bits))))),
        .f64 => @bitCast(bits),
    };
}

fn valueA(comptime element: Element, i: u32, k: u32) i32 {
    if (element == .b1) return @intFromBool((i * 7 + k * 3) % 5 < 2);
    const raw: i32 = @intCast((i * 3 + k * 5) % 7);
    return if (isUnsigned(element)) raw else raw - 3;
}

fn valueB(comptime element: Element, k: u32, j: u32) i32 {
    if (element == .b1) return @intFromBool((k * 5 + j * 11) % 3 == 0);
    const raw: i32 = @intCast((k * 2 + j * 7) % 5);
    return if (isUnsigned(element)) raw else raw - 2;
}

fn valueC(i: u32, j: u32, saturating: bool) i32 {
    const small = @as(i32, @intCast(i)) - @as(i32, @intCast(j));
    return if (saturating) std.math.maxInt(i32) - 40 + small else small;
}

fn product(comptime options: matrix.MmaOptions, i: u32, j: u32) i64 {
    var sum: i64 = 0;
    for (0..comptime options.shape.dimensions().k) |k| {
        const a = valueA(options.element, i, @intCast(k));
        const b = valueB(options.b_element orelse options.element, @intCast(k), j);
        sum += if (options.bit_operation) |operation| switch (operation) {
            .xor => a ^ b,
            .@"and" => a & b,
        } else a * b;
    }
    return sum;
}

fn expectedResult(
    comptime accumulator: Accumulator,
    sum: i64,
    saturating: bool,
) f64 {
    if (accumulator == .s32 and saturating) {
        return @floatFromInt(std.math.clamp(sum, std.math.minInt(i32), std.math.maxInt(i32)));
    }
    return @floatFromInt(sum);
}

fn note(first: u32, number: u32, passes: bool) u32 {
    return if (first == 0 and !passes) number else first;
}

fn report(out: [*]addrspace(.global) u32, first: u32) void {
    const earliest = nv.reduxSync(.min, u32, if (first == 0) 0xffffffff else first, .all);
    if (nv.laneid() == 0 and earliest != 0xffffffff) out[0] = earliest;
}

fn OperandRegisters(comptime element: Element, comptime elements: u32) type {
    if (element == .f64) return @Vector(elements, f64);
    return @Vector(elements * elementWidth(element) / 32, u32);
}

fn AccumulatorRegisters(comptime accumulator: Accumulator, comptime elements: u32) type {
    return switch (accumulator) {
        .f16 => @Vector(elements / 2, u32),
        .f32 => @Vector(elements, f32),
        .s32 => @Vector(elements, i32),
        .f64 => @Vector(elements, f64),
    };
}

fn elementsPerRegister(comptime element: Element) u32 {
    return if (element == .f64) 1 else 32 / elementWidth(element);
}

fn operandRegister(
    comptime options: matrix.MmaOptions,
    comptime operand: enum { a, b },
    register: u32,
    lane: u32,
) u64 {
    const b_element = options.b_element orelse options.element;
    const element = if (operand == .a) options.element else b_element;
    const per_register = elementsPerRegister(element);
    const width = if (element == .f64) 64 else elementWidth(element);
    const group = lane / 4;
    const thread = lane % 4;
    const rows_per_half = options.shape.dimensions().m / 8;
    var word: u64 = 0;
    for (0..per_register) |e| {
        const block = if (operand == .a) register / rows_per_half else register;
        const k: u32 = @intCast(block * 4 * per_register + thread * per_register + e);
        const value = if (operand == .a)
            valueA(element, group + 8 * (register % rows_per_half), k)
        else
            valueB(element, k, group);
        word |= encodeElement(element, value) << @intCast(e * width);
    }
    return word;
}

fn accumulatorPosition(element: u32, lane: u32) [2]u32 {
    return .{ lane / 4 + 8 * (element / 2), (lane % 4) * 2 + element % 2 };
}

fn mmaPasses(comptime options: matrix.MmaOptions) bool {
    const lane = nv.laneid();
    const size = comptime options.shape.dimensions();
    const b_element = options.b_element orelse options.element;
    const A = OperandRegisters(options.element, size.m * size.k / nv.warp_size);
    const B = OperandRegisters(b_element, size.k * size.n / nv.warp_size);
    const C = AccumulatorRegisters(options.accumulator, size.m * size.n / nv.warp_size);
    var a: A = undefined;
    var b: B = undefined;
    var c: C = undefined;
    inline for (0..@typeInfo(A).vector.len) |r| {
        const word = operandRegister(options, .a, r, lane);
        a[r] = if (options.element == .f64) @bitCast(word) else @truncate(word);
    }
    inline for (0..@typeInfo(B).vector.len) |r| {
        const word = operandRegister(options, .b, r, lane);
        b[r] = if (b_element == .f64) @bitCast(word) else @truncate(word);
    }
    const accumulator = options.accumulator;
    const saturating = options.satfinite;
    const length = @typeInfo(C).vector.len;
    inline for (0..length) |r| {
        var word: u64 = 0;
        if (accumulator == .f16) {
            const row: u32 = lane / 4 + 8 * @as(u32, r);
            for (0..2) |e| {
                const value = valueC(row, (lane % 4) * 2 + @as(u32, @intCast(e)), saturating);
                word |= encodeAccumulator(.f16, value) << @intCast(16 * e);
            }
        } else {
            const position = accumulatorPosition(r, lane);
            word = encodeAccumulator(accumulator, valueC(position[0], position[1], saturating));
        }
        c[r] = switch (@typeInfo(@typeInfo(C).vector.child)) {
            .float => @bitCast(@as(@Int(.unsigned, @bitSizeOf(@typeInfo(C).vector.child)), @truncate(word))),
            .int => @bitCast(@as(u32, @truncate(word))),
            else => unreachable,
        };
    }

    const d = nv.mmaSync(options, a, b, c);

    var passes = true;
    inline for (0..length) |r| {
        const bits: u64 = switch (@typeInfo(@typeInfo(C).vector.child)) {
            .float => @as(@Int(.unsigned, @bitSizeOf(@typeInfo(C).vector.child)), @bitCast(d[r])),
            .int => @as(u32, @bitCast(d[r])),
            else => unreachable,
        };
        if (accumulator == .f16) {
            for (0..2) |e| {
                const row: u32 = lane / 4 + 8 * @as(u32, r);
                const column = (lane % 4) * 2 + @as(u32, @intCast(e));
                const sum = product(options, row, column) + valueC(row, column, saturating);
                const got = decodeAccumulator(.f16, bits >> @intCast(16 * e));
                if (got != @as(f64, @floatFromInt(sum))) passes = false;
            }
        } else {
            const position = accumulatorPosition(r, lane);
            const sum = product(options, position[0], position[1]) +
                valueC(position[0], position[1], saturating);
            const expected = expectedResult(accumulator, sum, saturating);
            if (decodeAccumulator(accumulator, bits) != expected) passes = false;
        }
    }
    return passes;
}

export fn mma_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    comptime var number = 0;
    const shapes = [_]matrix.MmaSyncShape{ .m16n8k8, .m16n8k16 };
    inline for (shapes) |shape| {
        inline for (.{ Accumulator.f16, Accumulator.f32 }) |accumulator| {
            number += 1;
            first = note(first, number, mmaPasses(.{
                .shape = shape,
                .accumulator = accumulator,
                .element = .f16,
            }));
        }
        number += 1;
        first = note(first, number, mmaPasses(.{
            .shape = shape,
            .accumulator = .f32,
            .element = .bf16,
        }));
    }
    report(out, first);
}

export fn mma_tf32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    comptime var number = 0;
    inline for ([_]matrix.MmaSyncShape{ .m16n8k4, .m16n8k8 }) |shape| {
        number += 1;
        first = note(first, number, mmaPasses(.{
            .shape = shape,
            .accumulator = .f32,
            .element = .tf32,
        }));
    }
    report(out, first);
}

export fn mma_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    first = note(first, 1, mmaPasses(.{ .shape = .m8n8k4, .accumulator = .f64, .element = .f64 }));
    first = note(first, 2, mmaPasses(.{
        .shape = .m8n8k4,
        .accumulator = .f64,
        .element = .f64,
        .rounding = .rz,
    }));
    report(out, first);
}

export fn mma_int8(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    comptime var number = 0;
    const shapes = [_]matrix.MmaSyncShape{
        .m8n8k16,
        .m16n8k16,
        .m16n8k32,
    };
    inline for (shapes) |shape| {
        const pairs = .{
            .{ Element.s8, Element.s8 },
            .{ Element.u8, Element.u8 },
            .{ Element.s8, Element.u8 },
            .{ Element.u8, Element.s8 },
        };
        inline for (pairs) |types| {
            number += 1;
            first = note(first, number, mmaPasses(.{
                .shape = shape,
                .accumulator = .s32,
                .element = types[0],
                .b_element = types[1],
            }));
        }
        number += 1;
        first = note(first, number, mmaPasses(.{
            .shape = shape,
            .accumulator = .s32,
            .element = .s8,
            .b_element = .u8,
            .satfinite = true,
        }));
    }
    report(out, first);
}

export fn mma_int4(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    comptime var number = 0;
    const shapes = [_]matrix.MmaSyncShape{
        .m8n8k32,
        .m16n8k32,
        .m16n8k64,
    };
    inline for (shapes) |shape| {
        const pairs = .{
            .{ Element.s4, Element.s4 },
            .{ Element.u4, Element.u4 },
            .{ Element.s4, Element.u4 },
        };
        inline for (pairs) |types| {
            number += 1;
            first = note(first, number, mmaPasses(.{
                .shape = shape,
                .accumulator = .s32,
                .element = types[0],
                .b_element = types[1],
            }));
        }
        number += 1;
        first = note(first, number, mmaPasses(.{
            .shape = shape,
            .accumulator = .s32,
            .element = .s4,
            .b_element = .u4,
            .satfinite = true,
        }));
    }
    report(out, first);
}

export fn mma_single_bit(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    comptime var number = 0;
    const shapes = [_]matrix.MmaSyncShape{
        .m8n8k128,
        .m16n8k128,
        .m16n8k256,
    };
    inline for (shapes) |shape| {
        number += 1;
        first = note(first, number, mmaPasses(.{
            .shape = shape,
            .accumulator = .s32,
            .element = .b1,
            .bit_operation = .xor,
        }));
        if (comptime nv.gpu.ptx >= 71) {
            number += 1;
            first = note(first, number, mmaPasses(.{
                .shape = shape,
                .accumulator = .s32,
                .element = .b1,
                .bit_operation = .@"and",
            }));
        }
    }
    report(out, first);
}

fn elementPattern(matrix_index: u32, row: u32, column: u32) u32 {
    return matrix_index * 64 + row * 8 + column + 1;
}

fn fillLdmatrixSource(words: [*]addrspace(.shared) u32) void {
    const lane = nv.laneid();
    var word = lane;
    while (word < 4 * 8 * 4) : (word += nv.warp_size) {
        const matrix_index = word / 32;
        const row = word / 4 % 8;
        const column = word % 4 * 2;
        words[word] = elementPattern(matrix_index, row, column) |
            elementPattern(matrix_index, row, column + 1) << 16;
    }
    nv.barWarpSync(.all);
}

fn transposedFragment(matrix_index: u32, lane: u32) u32 {
    const row = lane % 4 * 2;
    return elementPattern(matrix_index, row, lane / 4) |
        elementPattern(matrix_index, row + 1, lane / 4) << 16;
}

fn straightFragment(matrix_index: u32, lane: u32) u32 {
    return elementPattern(matrix_index, lane / 4, lane % 4 * 2) |
        elementPattern(matrix_index, lane / 4, lane % 4 * 2 + 1) << 16;
}

export fn ldmatrix_loads(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const lane = nv.laneid();
    const warp = @workItemId(0) / nv.warp_size;
    const words = @as([*]addrspace(.shared) u32, &shared_words) + warp * region_words;
    fillLdmatrixSource(words);
    const row_address = &words[lane / 8 * 32 + lane % 8 * 4];
    var first: u32 = 0;
    comptime var number = 0;
    inline for (.{ matrix.LdmatrixCount.x1, .x2, .x4 }) |count| {
        const plain = nv.ldmatrix(count, false, row_address);
        const trans = nv.ldmatrix(count, true, row_address);
        inline for (0..@typeInfo(@TypeOf(plain)).vector.len) |i| {
            number += 1;
            first = note(first, number, plain[i] == straightFragment(i, lane));
            number += 1;
            first = note(first, number, trans[i] == transposedFragment(i, lane));
        }
    }
    report(out, first);
}

export fn movmatrix_transposes(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var first: u32 = 0;
    if (comptime nv.gpu.ptx >= 78) {
        const lane = nv.laneid();
        const moved = nv.movmatrix(straightFragment(0, lane));
        first = note(first, 1, moved == transposedFragment(0, lane));
    }
    report(out, first);
}
