const std = @import("std");
const nvptx = std.nvptx;

var bar: nvptx.Mbarrier addrspace(.shared) = undefined;
var tile: [64 * 64]f32 align(128) addrspace(.shared) = undefined;

fn smem(p: anytype) u32 {
    return @truncate(@intFromPtr(p));
}

fn laneId() u32 {
    return asm volatile ("mov.u32 %[r], %laneid;"
        : [r] "=r" (-> u32),
    );
}

/// Loads one 64x64 f32 tile with the Tensor Memory Accelerator (TMA) and sums it with a warp
/// reduction.
export fn tmaLoad(desc: u64, out: [*]addrspace(.global) f32, x: i32, y: i32) callconv(.kernel) void {
    const tid = @workItemId(0);
    if (tid == 0) {
        bar.init(1);
        nvptx.fenceProxyAsync(.shared);
        _ = bar.arriveExpectTx(64 * 64 * 4);
        asm volatile ("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[dst]], [%[desc], {%[x], %[y]}], [%[bar]];"
            :
            : [bar] "r" (smem(&bar)),
              [dst] "r" (smem(&tile)),
              [desc] "l" (desc),
              [x] "r" (x),
              [y] "r" (y),
            : .{ .memory = true });
    }
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
    while (!bar.tryWaitParity(0)) {}
    var sum: f32 = 0;
    var i: u32 = tid;
    while (i < 64 * 64) : (i += 32) sum += tile[i];
    inline for (.{ 16, 8, 4, 2, 1 }) |delta| {
        sum += asm volatile ("shfl.sync.bfly.b32 %[r], %[v], %[d], 0x1f, 0xffffffff;"
            : [r] "=f" (-> f32),
            : [v] "f" (sum),
              [d] "n" (delta),
        );
    }
    if (laneId() == 0) out[@workGroupId(0)] = sum;
}

/// Two outputs and an in-out operand.
export fn divmod(out: [*]addrspace(.global) u32, a: u32, b: u32) callconv(.kernel) void {
    var q: u32 = undefined;
    var r: u32 = undefined;
    var acc: u32 = a;
    asm ("div.u32 %[q], %[a], %[b];\n\trem.u32 %[r], %[a], %[b];\n\tadd.u32 %[acc], %[acc], 1;"
        : [q] "=r" (q),
          [r] "=r" (r),
          [acc] "+r" (acc),
        : [a] "r" (a),
          [b] "r" (b),
    );
    out[0] = q;
    out[1] = r;
    out[2] = acc;
}
