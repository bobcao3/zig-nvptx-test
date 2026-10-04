//! Integer tests that run unchanged on the host and on the GPU. Each test hashes the results of
//! one operation over many operand pairs and stores the hash in `out`.

const std = @import("std");

inline fn Unsigned(comptime T: type) type {
    return @Int(.unsigned, @typeInfo(T).int.bits);
}

inline fn ShiftType(comptime T: type) type {
    return @Int(.unsigned, @max(1, std.math.log2_int_ceil(u16, @typeInfo(T).int.bits)));
}

inline fn mix(h: u32, v: u64) u32 {
    var x = (h ^ @as(u32, @truncate(v))) *% 0x9E3779B1;
    x = (x ^ (x >> 15) ^ @as(u32, @truncate(v >> 32))) *% 0x85EBCA6B;
    return x ^ (x >> 13);
}

inline fn widen(x: anytype) u64 {
    return @as(Unsigned(@TypeOf(x)), @bitCast(x));
}

inline fn splitmix(seed: u64) u64 {
    var z = seed *% 0x9E3779B97F4A7C15 +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

inline fn edges(comptime T: type) [14]T {
    @setEvalBranchQuota(1000000);
    const bits = @typeInfo(T).int.bits;
    const top: u64 = @as(u64, 1) << (bits - 1);
    const half: u64 = @as(u64, 1) << (bits / 2);
    const list = [_]u64{ 0, 1, 2, 3, ~@as(u64, 0), ~@as(u64, 0) -% 1, top, top -% 1, top +% 1, half, half -% 1, half +% 1, 7, top >> 1 };
    var out: [14]T = undefined;
    for (list, 0..) |v, i| out[i] = @bitCast(@as(Unsigned(T), @truncate(v)));
    return out;
}

/// Random operands whose magnitude varies, so that both overflowing and non-overflowing cases occur.
inline fn randomValue(comptime T: type, seed: u64) T {
    const bits = @typeInfo(T).int.bits;
    const z = splitmix(seed);
    const shift: u6 = @truncate((z >> 58) % bits);
    const raw: Unsigned(T) = @truncate(splitmix(z) >> shift);
    return @bitCast(raw);
}

inline fn Pair(comptime T: type, comptime B: type) type {
    return struct { a: T, b: B };
}

const random_count = 3000;

/// The number of operand pairs for the type `T` and the right operand type `B`.
inline fn count(comptime T: type, comptime B: type) usize {
    const bits = @typeInfo(T).int.bits;
    if (B != T) {
        if (bits <= 8) return (1 << bits) * (1 << @typeInfo(B).int.bits);
        return edges(T).len * (1 << @typeInfo(B).int.bits) + random_count;
    }
    if (bits <= 8) return 1 << (2 * bits);
    return edges(T).len * edges(T).len + random_count;
}

inline fn pair(comptime T: type, comptime B: type, i: usize) Pair(T, B) {
    const bits = @typeInfo(T).int.bits;
    const e = comptime edges(T);
    if (B != T) {
        const b_bits = @typeInfo(B).int.bits;
        if (bits <= 8) {
            const a: Unsigned(T) = @truncate(i >> b_bits);
            return .{ .a = @bitCast(a), .b = @truncate(i) };
        }
        if (i < e.len * (1 << b_bits)) return .{ .a = e[i >> b_bits], .b = @truncate(i) };
        const j = i - e.len * (1 << b_bits);
        return .{ .a = randomValue(T, 2 * j), .b = @truncate(splitmix(j)) };
    }
    if (bits <= 8) {
        const a: Unsigned(T) = @truncate(i >> bits);
        const b: Unsigned(T) = @truncate(i);
        return .{ .a = @bitCast(a), .b = @bitCast(b) };
    }
    if (i < e.len * e.len) return .{ .a = e[i / e.len], .b = e[i % e.len] };
    const j = i - e.len * e.len;
    return .{ .a = randomValue(T, 2 * j), .b = randomValue(T, 2 * j + 1) };
}

const Op = enum { add, sub, mul, shl, add_sat, sub_sat, mul_sat, shl_sat };

inline fn RightType(comptime T: type, comptime op: Op) type {
    return switch (op) {
        .shl, .shl_sat => ShiftType(T),
        else => T,
    };
}

inline fn applyOverflow(comptime op: Op, a: anytype, b: anytype) @TypeOf(switch (op) {
    .add => @addWithOverflow(a, b),
    .sub => @subWithOverflow(a, b),
    .mul => @mulWithOverflow(a, b),
    .shl => @shlWithOverflow(a, b),
    else => unreachable,
}) {
    return switch (op) {
        .add => @addWithOverflow(a, b),
        .sub => @subWithOverflow(a, b),
        .mul => @mulWithOverflow(a, b),
        .shl => @shlWithOverflow(a, b),
        else => unreachable,
    };
}

inline fn applySat(comptime op: Op, a: anytype, b: anytype) @TypeOf(a) {
    return switch (op) {
        .add_sat => a +| b,
        .sub_sat => a -| b,
        .mul_sat => a *| b,
        .shl_sat => a <<| b,
        else => unreachable,
    };
}

/// Hashes the result and overflow bit of `@addWithOverflow` and its siblings, or the result of a
/// saturating operation.
inline fn hashBinary(comptime T: type, comptime op: Op) u32 {
    const Right = RightType(T, op);
    var h: u32 = 0;
    var i: usize = 0;
    while (i < count(T, Right)) : (i += 1) {
        const p = pair(T, Right, i);
        switch (op) {
            .add, .sub, .mul, .shl => {
                const r = applyOverflow(op, p.a, p.b);
                h = mix(h, widen(r[0]) ^ (@as(u64, r[1]) << 63));
            },
            else => h = mix(h, widen(applySat(op, p.a, p.b))),
        }
    }
    return h;
}

/// Hashes a saturating left shift whose right operand has the type `R`, which can be wider than
/// the shifted type and hold amounts that exceed its width.
inline fn hashShlSatWide(comptime T: type, comptime R: type) u32 {
    const bits = @typeInfo(T).int.bits;
    const amounts = [_]R{ 0, 1, 2, bits - 1, bits, bits + 1, 63, 64, 100, std.math.maxInt(R) / 2, std.math.maxInt(R) - 1, std.math.maxInt(R) };
    const e = comptime edges(T);
    var h: u32 = 0;
    var i: usize = 0;
    while (i < e.len * amounts.len) : (i += 1) {
        h = mix(h, widen(e[i / amounts.len] <<| amounts[i % amounts.len]));
    }
    return h;
}

const Unary = enum { clz, ctz, popcount, bit_reverse, byte_swap, abs };

inline fn UnaryResult(comptime T: type, comptime op: Unary) type {
    if (@typeInfo(T) == .vector) return @Vector(lanes, UnaryResult(@typeInfo(T).vector.child, op));
    return switch (op) {
        .clz, .ctz, .popcount => std.math.Log2IntCeil(T),
        .abs => Unsigned(T),
        else => T,
    };
}

inline fn unaryCount(comptime T: type) usize {
    const bits = @typeInfo(T).int.bits;
    if (bits <= 16) return 1 << bits;
    return edges(T).len + random_count;
}

inline fn unaryValue(comptime T: type, i: usize) T {
    const bits = @typeInfo(T).int.bits;
    if (bits <= 16) return @bitCast(@as(Unsigned(T), @truncate(i)));
    const e = comptime edges(T);
    if (i < e.len) return e[i];
    return randomValue(T, i);
}

inline fn applyUnary(comptime op: Unary, a: anytype) UnaryResult(@TypeOf(a), op) {
    return switch (op) {
        .clz => @clz(a),
        .ctz => @ctz(a),
        .popcount => @popCount(a),
        .bit_reverse => @bitReverse(a),
        .byte_swap => @byteSwap(a),
        .abs => @abs(a),
    };
}

inline fn hashUnary(comptime T: type, comptime op: Unary) u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i < unaryCount(T)) : (i += 1) h = mix(h, widen(applyUnary(op, unaryValue(T, i))));
    return h;
}

const lanes = 4;
const host = @import("builtin").cpu.arch != .nvptx64;

/// Applies the operation to vectors of `lanes` consecutive samples and hashes every lane.
inline fn hashVectorBinary(comptime T: type, comptime op: Op) u32 {
    const Right = RightType(T, op);
    var h: u32 = 0;
    var i: usize = 0;
    while (i + lanes <= count(T, Right)) : (i += lanes) {
        var a: @Vector(lanes, T) = undefined;
        var b: @Vector(lanes, Right) = undefined;
        inline for (0..lanes) |k| {
            const p = pair(T, Right, i + k);
            a[k] = p.a;
            b[k] = p.b;
        }
        switch (op) {
            .add, .sub, .mul, .shl => {
                const r = applyOverflow(op, a, b);
                inline for (0..lanes) |k| h = mix(h, widen(r[0][k]) ^ (@as(u64, r[1][k]) << 63));
            },
            else => {
                // The LLVM backend of Zig 0.17 fails on `@Vector(4, i24) <<| @Vector(4, u5)` in this
                // loop, so the host computes the reference result one lane at a time.
                if (host and op == .shl_sat) {
                    inline for (0..lanes) |k| h = mix(h, widen(a[k] <<| b[k]));
                } else {
                    const r = applySat(op, a, b);
                    inline for (0..lanes) |k| h = mix(h, widen(r[k]));
                }
            },
        }
    }
    return h;
}

inline fn hashVectorUnary(comptime T: type, comptime op: Unary) u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i + lanes <= unaryCount(T)) : (i += lanes) {
        var a: @Vector(lanes, T) = undefined;
        inline for (0..lanes) |k| a[k] = unaryValue(T, i + k);
        const r = applyUnary(op, a);
        inline for (0..lanes) |k| h = mix(h, widen(r[k]));
    }
    return h;
}

const all_types = .{ u1, i1, u7, i7, u8, i8, u15, i15, u16, i16, u17, i24, u31, i31, u32, i32, u33, i33, u63, i63, u64, i64 };
const vector_types = .{ u7, i8, u8, i16, u16, u17, i24, u32, i32, u33, u64, i64 };

inline fn fillBinary(out: anytype, comptime op: Op) void {
    @setEvalBranchQuota(1000000);
    inline for (all_types, 0..) |T, k| {
        if (@typeInfo(T).int.bits == 1 and (op == .shl or op == .shl_sat)) {
            out[k] = 0;
        } else {
            out[k] = hashBinary(T, op);
        }
    }
}

inline fn fillVectorBinary(out: anytype, comptime op: Op) void {
    @setEvalBranchQuota(1000000);
    inline for (vector_types, 0..) |T, k| out[k] = hashVectorBinary(T, op);
}

inline fn fillUnary(out: anytype, comptime op: Unary) void {
    @setEvalBranchQuota(1000000);
    var n: usize = 0;
    inline for (all_types) |T| {
        const bits = @typeInfo(T).int.bits;
        const signed = @typeInfo(T).int.signedness == .signed;
        const skip = switch (op) {
            .byte_swap => bits % 8 != 0,
            .abs => !signed or bits == 1,
            else => false,
        };
        if (!skip) {
            out[n] = hashUnary(T, op);
            n += 1;
        }
    }
}

inline fn fillVectorUnary(out: anytype, comptime op: Unary) void {
    @setEvalBranchQuota(1000000);
    var n: usize = 0;
    inline for (vector_types) |T| {
        const bits = @typeInfo(T).int.bits;
        const signed = @typeInfo(T).int.signedness == .signed;
        const skip = switch (op) {
            .byte_swap => bits % 8 != 0,
            .abs => !signed,
            else => false,
        };
        if (!skip) {
            out[n] = hashVectorUnary(T, op);
            n += 1;
        }
    }
}

pub fn add(out: anytype) void {
    fillBinary(out, .add);
}
pub fn sub(out: anytype) void {
    fillBinary(out, .sub);
}
pub fn mul(out: anytype) void {
    fillBinary(out, .mul);
}
pub fn shl(out: anytype) void {
    fillBinary(out, .shl);
}
pub fn add_sat(out: anytype) void {
    fillBinary(out, .add_sat);
}
pub fn sub_sat(out: anytype) void {
    fillBinary(out, .sub_sat);
}
pub fn mul_sat(out: anytype) void {
    fillBinary(out, .mul_sat);
}
pub fn shl_sat(out: anytype) void {
    fillBinary(out, .shl_sat);
}
pub inline fn shl_sat_wide(out: anytype) void {
    @setEvalBranchQuota(1000000);
    var n: usize = 0;
    inline for (all_types) |T| {
        inline for (.{ u8, u32, u64 }) |R| {
            out[n] = hashShlSatWide(T, R);
            n += 1;
        }
    }
}
pub fn clz(out: anytype) void {
    fillUnary(out, .clz);
}
pub fn ctz(out: anytype) void {
    fillUnary(out, .ctz);
}
pub fn popcount(out: anytype) void {
    fillUnary(out, .popcount);
}
pub fn bit_reverse(out: anytype) void {
    fillUnary(out, .bit_reverse);
}
pub fn byte_swap(out: anytype) void {
    fillUnary(out, .byte_swap);
}
pub fn abs(out: anytype) void {
    fillUnary(out, .abs);
}
pub fn vec_add(out: anytype) void {
    fillVectorBinary(out, .add);
}
pub fn vec_sub(out: anytype) void {
    fillVectorBinary(out, .sub);
}
pub fn vec_mul(out: anytype) void {
    fillVectorBinary(out, .mul);
}
pub fn vec_shl(out: anytype) void {
    fillVectorBinary(out, .shl);
}
pub fn vec_add_sat(out: anytype) void {
    fillVectorBinary(out, .add_sat);
}
pub fn vec_sub_sat(out: anytype) void {
    fillVectorBinary(out, .sub_sat);
}
pub fn vec_mul_sat(out: anytype) void {
    fillVectorBinary(out, .mul_sat);
}
pub fn vec_shl_sat(out: anytype) void {
    fillVectorBinary(out, .shl_sat);
}
pub fn vec_clz(out: anytype) void {
    fillVectorUnary(out, .clz);
}
pub fn vec_ctz(out: anytype) void {
    fillVectorUnary(out, .ctz);
}
pub fn vec_popcount(out: anytype) void {
    fillVectorUnary(out, .popcount);
}
pub fn vec_bit_reverse(out: anytype) void {
    fillVectorUnary(out, .bit_reverse);
}
pub fn vec_byte_swap(out: anytype) void {
    fillVectorUnary(out, .byte_swap);
}
pub fn vec_abs(out: anytype) void {
    fillVectorUnary(out, .abs);
}

pub const names = [_][]const u8{
    "add",         "sub",          "mul",             "shl",           "add_sat",     "sub_sat",
    "mul_sat",     "shl_sat",      "shl_sat_wide",    "clz",           "ctz",         "popcount",
    "bit_reverse", "byte_swap",    "abs",             "vec_add",       "vec_sub",     "vec_mul",
    "vec_shl",     "vec_add_sat",  "vec_sub_sat",     "vec_mul_sat",   "vec_shl_sat", "vec_clz",
    "vec_ctz",     "vec_popcount", "vec_bit_reverse", "vec_byte_swap", "vec_abs",
};
