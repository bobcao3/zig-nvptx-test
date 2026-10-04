const Pair = struct { a: u32, b: u64 };
const E = error{ Odd, Big };

noinline fn swap(p: Pair) Pair {
    return .{ .a = @truncate(p.b), .b = p.a };
}

noinline fn half(x: u32) E!u32 {
    if (x & 1 != 0) return error.Odd;
    if (x > 1000) return error.Big;
    return x / 2;
}

noinline fn find(s: []const u32, v: u32) ?usize {
    for (s, 0..) |e, i| if (e == v) return i;
    return null;
}

noinline fn sum(s: []const u8) u32 {
    var t: u32 = 0;
    for (s) |c| t += c;
    return t;
}

fn classify(x: u32) u32 {
    return switch (x) {
        0 => 10,
        1, 2 => 20,
        3...7 => 30,
        else => 40,
    };
}

export fn calls(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const p = swap(.{ .a = 7, .b = 9 });
    out[0] = p.a;
    out[1] = @truncate(p.b);
    out[2] = half(10) catch 99;
    out[3] = half(11) catch |err| if (err == error.Odd) 1 else 2;
    out[4] = half(2000) catch |err| if (err == error.Big) 3 else 4;
    const arr = [_]u32{ 5, 6, 7, 8 };
    out[5] = @intCast(find(&arr, 7) orelse 100);
    out[6] = @intCast(find(&arr, 9) orelse 100);
    out[7] = sum("hello");
    var acc: u32 = 0;
    for (0..9) |i| acc += classify(@intCast(i));
    out[8] = acc;
    var x: i32 = -7;
    out[9] = @bitCast(@divFloor(x, 2));
    out[10] = @bitCast(@mod(x, 3));
    x = @rem(x, 3);
    out[11] = @bitCast(x);
    const f: f32 = 2.5;
    out[12] = @bitCast(@sqrt(f * f));
    out[13] = @intFromFloat(@floor(f) + @ceil(f));
    var v: @Vector(4, u32) = .{ 1, 2, 3, 4 };
    v = v * @as(@Vector(4, u32), @splat(3)) + v;
    out[14] = @reduce(.Add, v);
    out[15] = @popCount(@as(u32, 0xff00ff)) + @clz(@as(u32, 1)) + @ctz(@as(u32, 0x80));
}
