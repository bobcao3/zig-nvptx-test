export fn vec(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var v: @Vector(4, u32) = .{ 1, 2, 3, 4 };
    v = v * @as(@Vector(4, u32), @splat(3)) + v;
    out[0] = @reduce(.Add, v);
}
