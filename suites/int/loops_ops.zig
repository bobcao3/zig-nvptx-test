//! Loop tests that run unchanged on the host and on the GPU. Each test hashes the iteration counts
//! and the final values of loops and stores the hash in `out`. The loops exit on a float
//! comparison with NaN operands, on an integer comparison at the edges of the integer range, or
//! after they read an array that a function takes by value at a run-time index.

const std = @import("std");

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

/// Every NaN hashes to the same value, because the host and the GPU can differ in the NaN payload.
inline fn bits(x: anytype) u64 {
    if (@typeInfo(@TypeOf(x)) == .float and std.math.isNan(x)) return 0x7FC00000;
    return @as(@Int(.unsigned, @bitSizeOf(@TypeOf(x))), @bitCast(x));
}

const float_values = [_]f64{ 0, -0.0, 0.5, 1, -1, 2.5, 7, 100, 65504, -65504, 1e30, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) };
const max_iterations = 6;

inline fn floatLoops(comptime T: type, comptime op: std.math.CompareOperator, comptime negated: bool) u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i < float_values.len * float_values.len) : (i += 1) {
        var x: T = @floatCast(float_values[i / float_values.len]);
        const limit: T = @floatCast(float_values[i % float_values.len]);
        var n: u32 = 0;
        while (negated != std.math.compare(x, op, limit)) : (x += 1.5) {
            n += 1;
            if (n == max_iterations) break;
        }
        h = mix(mix(h, n), bits(x));
    }
    return h;
}

inline fn floatOps(comptime T: type) u32 {
    var h: u32 = 0;
    inline for (.{ .lt, .lte, .gt, .gte, .eq, .neq }) |op| {
        h = mix(h, floatLoops(T, op, false));
        h = mix(h, floatLoops(T, op, true));
    }
    return h;
}

inline fn intLoops(comptime T: type, comptime op: std.math.CompareOperator) u32 {
    const edges = [_]T{ std.math.minInt(T), std.math.minInt(T) + 1, 0, 1, 2, std.math.maxInt(T) - 1, std.math.maxInt(T) };
    var h: u32 = 0;
    var i: usize = 0;
    while (i < edges.len * edges.len) : (i += 1) {
        var x = edges[i / edges.len];
        const limit = edges[i % edges.len];
        var n: u32 = 0;
        while (std.math.compare(x, op, limit)) : (x +%= 3) {
            n += 1;
            if (n == max_iterations) break;
        }
        h = mix(mix(h, n), bits(x));
    }
    return h;
}

inline fn intOps(comptime T: type) u32 {
    var h: u32 = 0;
    inline for (.{ .lt, .lte, .gt, .gte, .eq, .neq }) |op| h = mix(h, intLoops(T, op));
    return h;
}

const Pair = extern struct { index: u32, weight: f32 };

inline fn element(comptime T: type, seed: u64) T {
    const z = splitmix(seed);
    return switch (@typeInfo(T)) {
        .int => @truncate(z),
        .float => @floatFromInt(z % 1000),
        .@"struct" => .{ .index = @truncate(z), .weight = @floatFromInt((z >> 32) % 1000) },
        else => unreachable,
    };
}

inline fn hashElement(h: u32, e: anytype) u32 {
    if (@typeInfo(@TypeOf(e)) == .@"struct") return mix(mix(h, e.index), bits(e.weight));
    return mix(h, bits(e));
}

inline fn pick(row: anytype, index: usize) @typeInfo(@TypeOf(row)).array.child {
    return row[index];
}

inline fn arrayLoops(comptime T: type, comptime len: usize) u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        var source: [len]T = undefined;
        for (&source, 0..) |*e, k| e.* = element(T, i * len + k);
        const row = source;
        var x = splitmix(i);
        var k: usize = 0;
        while (k < 40) : (k += 1) {
            x = splitmix(x);
            h = hashElement(h, pick(row, x % len));
        }
        for (row) |e| h = hashElement(h, e);
    }
    return h;
}

inline fn sentinelLoops() u32 {
    var h: u32 = 0;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        var source: [7:0]u32 = @splat(0);
        for (&source, 0..) |*e, k| e.* = element(u32, i * 7 + k) | 1;
        const row = source;
        h = mix(h, pick(row, splitmix(i) % 8));
        h = mix(h, pick(row, (i + 7) % 8));
    }
    return h;
}

pub inline fn float_f16(out: anytype) void {
    out[0] = floatOps(f16);
}
pub inline fn float_f32(out: anytype) void {
    out[0] = floatOps(f32);
}
pub inline fn float_f64(out: anytype) void {
    out[0] = floatOps(f64);
}
pub inline fn int_i8(out: anytype) void {
    out[0] = intOps(i8);
}
pub inline fn int_u32(out: anytype) void {
    out[0] = intOps(u32);
}
pub inline fn int_i32(out: anytype) void {
    out[0] = intOps(i32);
}
pub inline fn int_u64(out: anytype) void {
    out[0] = intOps(u64);
}
pub inline fn array_u32_8(out: anytype) void {
    out[0] = arrayLoops(u32, 8);
}
pub inline fn array_u32_32(out: anytype) void {
    out[0] = arrayLoops(u32, 32);
}
pub inline fn array_u32_33(out: anytype) void {
    out[0] = arrayLoops(u32, 33);
}
pub inline fn array_u8_32(out: anytype) void {
    out[0] = arrayLoops(u8, 32);
}
pub inline fn array_u8_33(out: anytype) void {
    out[0] = arrayLoops(u8, 33);
}
pub inline fn array_f64_16(out: anytype) void {
    out[0] = arrayLoops(f64, 16);
}
pub inline fn array_f64_17(out: anytype) void {
    out[0] = arrayLoops(f64, 17);
}
pub inline fn array_pair_16(out: anytype) void {
    out[0] = arrayLoops(Pair, 16);
}
pub inline fn array_pair_17(out: anytype) void {
    out[0] = arrayLoops(Pair, 17);
}
pub inline fn array_sentinel(out: anytype) void {
    out[0] = sentinelLoops();
}

pub const names = [_][]const u8{
    "float_f16",    "float_f32",    "float_f64",     "int_i8",        "int_u32",        "int_i32",
    "int_u64",      "array_u32_8",  "array_u32_32",  "array_u32_33",  "array_u8_32",    "array_u8_33",
    "array_f64_16", "array_f64_17", "array_pair_16", "array_pair_17", "array_sentinel",
};
