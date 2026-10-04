/// A TMEM address that `tcgen05.alloc` returns.
const TmemAddress = enum(u32) { _ };

const MatrixDescriptor = packed struct(u64) {
    start: u14,
    unused0: u2 = 0,
    leading: u14,
    unused1: u2 = 0,
    stride: u14,
    unused2: u2 = 0,
    base_offset: u3 = 0,
    unused3: u10 = 0,
    swizzle: u3,
};

var slot: u32 addrspace(.shared) = undefined;
var a_tile: [64 * 16]u16 align(1024) addrspace(.shared) = undefined;

/// wgmma m64n128k16 with a [64]f32 accumulator as one register list.
export fn gemm(out: [*]addrspace(.global) f32, b_desc: MatrixDescriptor) callconv(.kernel) void {
    var acc: [64]f32 = @splat(0);
    const a_desc: MatrixDescriptor = .{
        .start = @truncate(@intFromPtr(&a_tile) >> 4),
        .leading = 1,
        .stride = 64,
        .swizzle = 1,
    };
    asm volatile ("wgmma.fence.sync.aligned;" ::: .{ .memory = true });
    asm volatile ("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 %[d], %[a], %[b], 1, 1, 1, 0, 0;"
        : [d] "+f" (acc),
        : [a] "l" (a_desc),
          [b] "l" (b_desc),
        : .{ .memory = true });
    asm volatile ("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;" ::: .{ .memory = true });
    inline for (acc, 0..) |v, i| out[@workItemId(0) * 64 + i] = v;
}
