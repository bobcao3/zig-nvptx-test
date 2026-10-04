//! Each kernel compares `@reduce` at run time with the result that the compiler computes at compile
//! time, for vectors of every width and element kind. A failed check stores its line number in
//! `out[0]`. Two NaN results count as equal, because their payloads may differ.

const std = @import("std");

const Out = [*]addrspace(.global) u32;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

noinline fn hide(comptime V: type, v: V) V {
    return v;
}

fn same(comptime T: type, a: T, b: T) bool {
    if (@typeInfo(T) == .float) {
        if (std.math.isNan(a) or std.math.isNan(b)) return std.math.isNan(a) and std.math.isNan(b);
        const Bits = @Int(.unsigned, @bitSizeOf(T));
        return @as(Bits, @bitCast(a)) == @as(Bits, @bitCast(b));
    }
    return a == b;
}

fn check(out: Out, line: u32, comptime op: std.lang.ReduceOp, comptime V: type, comptime v: V) void {
    const E = @typeInfo(V).vector.child;
    const want = comptime @reduce(op, v);
    expect(out, line, same(E, @reduce(op, hide(V, v)), want));
}

const nan32 = std.math.nan(f32);

export fn float_nan_and_zero(out: Out) callconv(.kernel) void {
    check(out, @src().line, .Min, @Vector(7, f32), @splat(nan32));
    check(out, @src().line, .Max, @Vector(7, f32), @splat(nan32));
    check(out, @src().line, .Min, @Vector(7, f32), .{ nan32, 3, nan32, -2, 5, nan32, 1 });
    check(out, @src().line, .Max, @Vector(5, f64), .{ std.math.nan(f64), -1, -0.0, 0.0, -7 });
    check(out, @src().line, .Add, @Vector(6, f32), @splat(-0.0));
    check(out, @src().line, .Add, @Vector(3, f32), @splat(-0.0));
    check(out, @src().line, .Mul, @Vector(6, f32), .{ -1, 2, -0.5, 4, 1, -1 });
    check(out, @src().line, .Min, @Vector(8, f16), .{ 3, -1, std.math.nan(f16), 2, -0.0, 0.0, 7, 1 });
    check(out, @src().line, .Min, @Vector(1, f32), .{nan32});
}

export fn float_order(out: Out) callconv(.kernel) void {
    check(out, @src().line, .Add, @Vector(9, f64), .{ 1e16, 1, -1e16, 1, 1, 0.5, -0.25, 3, 1e-3 });
    check(out, @src().line, .Add, @Vector(8, f32), .{ 1e8, 1, -1e8, 1, 2, 3, 4, 5 });
    check(out, @src().line, .Mul, @Vector(5, f64), .{ 1e200, 1e200, 1e-200, 1e-200, 3 });
    check(out, @src().line, .Add, @Vector(4, f16), .{ 2048, 1, 1, 1 });
}

export fn integers(out: Out) callconv(.kernel) void {
    check(out, @src().line, .Add, @Vector(40, u32), std.simd.iota(u32, 40) *% @as(@Vector(40, u32), @splat(0x9e37_79b9)));
    check(out, @src().line, .Min, @Vector(40, i32), std.simd.iota(i32, 40) - @as(@Vector(40, i32), @splat(20)));
    check(out, @src().line, .Xor, @Vector(17, u12), std.simd.iota(u12, 17) *% @as(@Vector(17, u12), @splat(0x9d)));
    check(out, @src().line, .Max, @Vector(5, i7), .{ -64, 63, 0, -1, 5 });
    check(out, @src().line, .And, @Vector(5, bool), .{ true, true, false, true, true });
    check(out, @src().line, .Or, @Vector(5, bool), .{ false, false, false, true, false });
    check(out, @src().line, .Xor, @Vector(5, bool), .{ true, true, false, true, true });
}

export fn wide_integers(out: Out) callconv(.kernel) void {
    const big: u128 = 0xfedc_ba98_7654_3210_0123_4567_89ab_cdef;
    check(out, @src().line, .Add, @Vector(3, u128), .{ big, big, 1 });
    check(out, @src().line, .Mul, @Vector(3, u128), .{ big, 3, big });
    check(out, @src().line, .Min, @Vector(4, i100), .{ -(1 << 98), 5, 1 << 90, -3 });
    check(out, @src().line, .Max, @Vector(4, i100), .{ -(1 << 98), 5, 1 << 90, -3 });
    check(out, @src().line, .Xor, @Vector(3, u65), .{ 1 << 64, 3, 5 });
}
