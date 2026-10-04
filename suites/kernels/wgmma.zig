/// wgmma m64n16k16. The accumulator is one register-list operand.
export fn wgmma(out: [*]addrspace(.global) f32, da: u64, db: u64) callconv(.kernel) void {
    var d: @Vector(8, f32) = @splat(0);
    asm volatile ("wgmma.fence.sync.aligned;" ::: .{ .memory = true });
    asm volatile ("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 %[d], %[a], %[b], 1, 1, 1, 0, 0;"
        : [d] "+f" (d),
        : [a] "l" (da),
          [b] "l" (db),
        : .{ .memory = true });
    asm volatile ("wgmma.commit_group.sync.aligned;\n\twgmma.wait_group.sync.aligned 0;" ::: .{ .memory = true });
    const arr: [8]f32 = d;
    for (arr, 0..) |v, i| out[@workItemId(0) * 8 + i] = v;
}
