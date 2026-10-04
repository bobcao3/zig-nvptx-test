//! Floating-point tests that run unchanged on the host and on the GPU. Each test hashes the
//! results of one operation over many operands and stores the hash in `out`.

const std = @import("std");

inline fn Bits(comptime T: type) type {
    return @Int(.unsigned, @bitSizeOf(T));
}

inline fn mix(h: u32, v: u64) u32 {
    var x = (h ^ @as(u32, @truncate(v))) *% 0x9E3779B1;
    x = (x ^ (x >> 15) ^ @as(u32, @truncate(v >> 32))) *% 0x85EBCA6B;
    return x ^ (x >> 13);
}

inline fn splitmix(seed: u64) u64 {
    var z = seed *% 0x9E3779B97F4A7C15 +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// The bits of `x`. Every NaN has the same result unless `exact` is set.
inline fn hashBits(x: anytype, comptime exact: bool) u64 {
    if (!exact and std.math.isNan(x)) return 0x7FC00000;
    return @as(Bits(@TypeOf(x)), @bitCast(x));
}

const specials = [_]f64{
    0.0,               -0.0,                0.5,               -0.5,   0.25,    1.5,                -1.5,      2.5,
    -2.5,              0.49999999999999994, 1.0,               -1.0,   3.0,     7.5,                100.5,     -100.5,
    1e10,              -1e10,               1e30,              3.4e38, 65504.0, -65504.0,           6.0e-8,    1e-45,
    1e-310,            2.0,                 -2.0,              0.1,    -0.1,    4503599627370495.5, 8388607.5, 2047.5,
    std.math.inf(f64), -std.math.inf(f64),  std.math.nan(f64), 6.1e-5, 5.96e-8, 1.0e300,            4.9e-324,  9007199254740993.0,
};

/// Operands that include special values, ties, and values near ties, and random bit patterns.
inline fn sample(comptime T: type, i: usize) T {
    @setEvalBranchQuota(1000000);
    if (i < specials.len) return @floatCast(specials[i]);
    const j = i - specials.len;
    const z = splitmix(j);
    const bits = @bitSizeOf(T);
    const mantissa = std.math.floatMantissaBits(T);
    const exponent = std.math.floatExponentBits(T);
    const bias = (1 << (exponent - 1)) - 1;
    switch (j % 4) {
        0 => return @bitCast(@as(Bits(T), @truncate(splitmix(z)))),
        1 => {
            const eighths: i64 = @as(i64, @intCast(z % 4001)) - 2000;
            return @as(T, @floatFromInt(eighths)) / 8;
        },
        2 => {
            const tie: T = @as(T, @floatFromInt((z >> 8) % 64)) + 0.5;
            const adjust: i64 = @as(i64, @intCast((z >> 20) % 3)) - 1;
            const moved: Bits(T) = @intCast(@as(i64, @intCast(@as(Bits(T), @bitCast(tie)))) + adjust);
            return @bitCast(moved);
        },
        else => {
            const span: u64 = @min(40, bias);
            const field: Bits(T) = @intCast(bias - span / 2 + (z >> 12) % span);
            const keep: Bits(T) = (@as(Bits(T), 1) << (bits - 1)) | ((@as(Bits(T), 1) << mantissa) - 1);
            return @bitCast((@as(Bits(T), @truncate(splitmix(z))) & keep) | (field << mantissa));
        },
    }
}

inline fn count(comptime T: type) usize {
    return switch (@bitSizeOf(T)) {
        16 => 1 << 16,
        else => 6000,
    };
}

/// The i-th operand pair. Half precision uses every bit pattern as the left operand.
inline fn pair(comptime T: type, i: usize) [2]T {
    @setEvalBranchQuota(1000000);
    if (@bitSizeOf(T) == 16) return .{ @bitCast(@as(u16, @truncate(i))), sample(T, splitmix(i) % 4096) };
    return .{ sample(T, i), sample(T, splitmix(i) % 3000) };
}

const Op = enum { round, rem, mod, neg, abs };

inline fn apply(comptime op: Op, a: anytype, b: anytype) @TypeOf(a) {
    return switch (op) {
        .round => @round(a),
        .rem => @rem(a, b),
        .mod => @mod(a, b),
        .neg => -a,
        .abs => @abs(a),
    };
}

inline fn divisor(comptime op: Op, b: anytype) @TypeOf(b) {
    if (op != .mod) return b;
    const d = @abs(b);
    return if (d == 0) 1 else d;
}

inline fn hashScalar(comptime T: type, comptime op: Op) u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i < count(T)) : (i += 1) {
        const p = pair(T, i);
        h = mix(h, hashBits(apply(op, p[0], divisor(op, p[1])), op == .neg or op == .abs));
    }
    return h;
}

const lanes = 4;

inline fn hashVector(comptime T: type, comptime op: Op) u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i + lanes <= count(T)) : (i += lanes) {
        var a: @Vector(lanes, T) = undefined;
        var b: @Vector(lanes, T) = undefined;
        inline for (0..lanes) |k| {
            const p = pair(T, i + k);
            a[k] = p[0];
            b[k] = divisor(op, p[1]);
        }
        const r = apply(op, a, b);
        inline for (0..lanes) |k| h = mix(h, hashBits(r[k], op == .neg or op == .abs));
    }
    return h;
}

pub inline fn round_f16(out: anytype) void {
    out[0] = hashScalar(f16, .round);
}
pub inline fn round_f32(out: anytype) void {
    out[0] = hashScalar(f32, .round);
}
pub inline fn round_f64(out: anytype) void {
    out[0] = hashScalar(f64, .round);
}
pub inline fn rem_f16(out: anytype) void {
    out[0] = hashScalar(f16, .rem);
}
pub inline fn rem_f32(out: anytype) void {
    out[0] = hashScalar(f32, .rem);
}
pub inline fn rem_f64(out: anytype) void {
    out[0] = hashScalar(f64, .rem);
}
pub inline fn mod_f16(out: anytype) void {
    out[0] = hashScalar(f16, .mod);
}
pub inline fn mod_f32(out: anytype) void {
    out[0] = hashScalar(f32, .mod);
}
pub inline fn mod_f64(out: anytype) void {
    out[0] = hashScalar(f64, .mod);
}
pub inline fn neg_f16(out: anytype) void {
    out[0] = hashScalar(f16, .neg);
}
pub inline fn neg_f32(out: anytype) void {
    out[0] = hashScalar(f32, .neg);
}
pub inline fn neg_f64(out: anytype) void {
    out[0] = hashScalar(f64, .neg);
}
pub inline fn abs_f16(out: anytype) void {
    out[0] = hashScalar(f16, .abs);
}
pub inline fn abs_f32(out: anytype) void {
    out[0] = hashScalar(f32, .abs);
}
pub inline fn abs_f64(out: anytype) void {
    out[0] = hashScalar(f64, .abs);
}
pub inline fn vec_round_f16(out: anytype) void {
    out[0] = hashVector(f16, .round);
}
pub inline fn vec_round_f32(out: anytype) void {
    out[0] = hashVector(f32, .round);
}
pub inline fn vec_round_f64(out: anytype) void {
    out[0] = hashVector(f64, .round);
}
pub inline fn vec_rem_f16(out: anytype) void {
    out[0] = hashVector(f16, .rem);
}
pub inline fn vec_rem_f32(out: anytype) void {
    out[0] = hashVector(f32, .rem);
}
pub inline fn vec_rem_f64(out: anytype) void {
    out[0] = hashVector(f64, .rem);
}
pub inline fn vec_mod_f16(out: anytype) void {
    out[0] = hashVector(f16, .mod);
}
pub inline fn vec_mod_f32(out: anytype) void {
    out[0] = hashVector(f32, .mod);
}
pub inline fn vec_mod_f64(out: anytype) void {
    out[0] = hashVector(f64, .mod);
}
pub inline fn vec_neg_f16(out: anytype) void {
    out[0] = hashVector(f16, .neg);
}
pub inline fn vec_neg_f32(out: anytype) void {
    out[0] = hashVector(f32, .neg);
}
pub inline fn vec_abs_f16(out: anytype) void {
    out[0] = hashVector(f16, .abs);
}

pub const names = [_][]const u8{
    "round_f16",   "round_f32",   "round_f64",   "rem_f16",       "rem_f32",       "rem_f64",
    "mod_f16",     "mod_f32",     "mod_f64",     "neg_f16",       "neg_f32",       "neg_f64",
    "abs_f16",     "abs_f32",     "abs_f64",     "vec_round_f16", "vec_round_f32", "vec_round_f64",
    "vec_rem_f16", "vec_rem_f32", "vec_rem_f64", "vec_mod_f16",   "vec_mod_f32",   "vec_mod_f64",
    "vec_neg_f16", "vec_neg_f32", "vec_abs_f16",
};
