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

/// tcgen05: allocate TMEM, load 128 columns, free.
export fn tmem(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    asm volatile ("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%[slot]], 128;"
        :
        : [slot] "r" (&slot),
        : .{ .memory = true });
    const addr: TmemAddress = @enumFromInt(slot);
    var regs: [128]u32 = undefined;
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x128.b32 %[r], [%[addr]];\ntcgen05.wait::ld.sync.aligned;"
        : [r] "=r" (regs),
        : [addr] "r" (addr),
        : .{ .memory = true });
    inline for (regs, 0..) |v, i| out[@workItemId(0) * 128 + i] = v;
    asm volatile ("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %[addr], 128;\ntcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
        :
        : [addr] "r" (addr),
        : .{ .memory = true });
}
