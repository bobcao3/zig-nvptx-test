export fn saxpy(
    n: u32,
    a: f32,
    x: [*]addrspace(.global) const f32,
    y: [*]addrspace(.global) f32,
) callconv(.kernel) void {
    const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    if (i < n) y[i] = a * x[i] + y[i];
}
