const std = @import("std");

pub const TensorMap = extern struct {
    opaque_bytes: [128]u8 align(64),
};

var bar: std.nvptx.Mbarrier addrspace(.shared) = undefined;
var tile: [32 * 16]f32 align(128) addrspace(.shared) = undefined;

/// Loads the tile at (x, y) of a 2D f32 tensor and writes its 32 * 16 words to `out`.
export fn tmaTile(map: TensorMap, out: [*]addrspace(.global) f32, x: i32, y: i32) callconv(.kernel) void {
    const tid = @workItemId(0);
    if (tid == 0) {
        bar.init(1);
        std.nvptx.fenceProxyAsync(.shared);
        _ = bar.arriveExpectTx(32 * 16 * 4);
        asm volatile ("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[dst]], [%[map], {%[x], %[y]}], [%[bar]];"
            :
            : [bar] "r" (@as(u32, @truncate(@intFromPtr(&bar)))),
              [dst] "r" (@as(u32, @truncate(@intFromPtr(&tile)))),
              [map] "l" (&map),
              [x] "r" (x),
              [y] "r" (y),
            : .{ .memory = true });
    }
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
    while (!bar.tryWaitParity(0)) {}
    var i = tid;
    while (i < 32 * 16) : (i += @workGroupSize(0)) out[i] = tile[i];
}
