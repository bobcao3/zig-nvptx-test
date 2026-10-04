const std = @import("std");
const nvptx = std.nvptx;

const Out = [*]addrspace(.global) u32;

fn fail(out: Out, check: u32) void {
    var first = @atomicLoad(u32, &out[0], .monotonic);
    while (first == 0 or check < first) {
        first = @cmpxchgWeak(u32, &out[0], first, check, .monotonic, .monotonic) orelse return;
    }
}

fn expect(out: Out, check: u32, ok: bool) void {
    if (!ok) fail(out, check);
}

export fn sreg_ranges(out: Out) callconv(.kernel) void {
    expect(out, 1, nvptx.smid() < nvptx.nsmid());
    expect(out, 2, nvptx.warpid() < nvptx.nwarpid());
    expect(out, 3, nvptx.totalSmemSize() >= nvptx.dynamicSmemSize());
    expect(out, 4, nvptx.gridid() != 0);
    expect(out, 5, nvptx.laneid() < nvptx.warp_size);
    _ = nvptx.clockHi();
    _ = nvptx.envreg(0);
}

export fn sreg_lanemasks(out: Out) callconv(.kernel) void {
    const lane: u5 = @intCast(nvptx.laneid());
    const eq = nvptx.lanemaskEq();
    const lt = nvptx.lanemaskLt();
    const le = nvptx.lanemaskLe();
    const gt = nvptx.lanemaskGt();
    const ge = nvptx.lanemaskGe();
    expect(out, 1, eq == @as(u32, 1) << lane);
    expect(out, 2, lt == eq -% 1);
    expect(out, 3, le == lt | eq);
    expect(out, 4, gt == ~le);
    expect(out, 5, ge == ~lt);
    expect(out, 6, @popCount(lt) == lane);
}

export fn sreg_globaltimer_monotonic(out: Out) callconv(.kernel) void {
    var previous = nvptx.globaltimer();
    for (0..64) |_| {
        nvptx.nanosleep(100);
        const now = nvptx.globaltimer();
        expect(out, 1, now >= previous);
        previous = now;
    }
    const before = nvptx.globaltimer();
    const low = nvptx.globaltimerLo();
    const high = nvptx.globaltimerHi();
    const after = nvptx.globaltimer();
    const combined = @as(u64, high) << 32 | low;
    expect(out, 2, before >> 32 <= high and high <= after >> 32);
    expect(out, 3, combined >= before -% 0xffff_ffff and combined <= after +% 0xffff_ffff);
    expect(out, 4, after >= before);
}

export fn sreg_nanosleep_returns(out: Out) callconv(.kernel) void {
    const start = nvptx.clock64();
    nvptx.nanosleep(0);
    nvptx.nanosleep(1);
    nvptx.nanosleep(10_000);
    expect(out, 1, nvptx.clock64() > start);
    nvptx.pmevent(0);
    nvptx.pmeventMask(0x3);
}

var shared_pointers: [128]?*u32 addrspace(.shared) = undefined;

export fn sreg_malloc_free(out: Out) callconv(.kernel) void {
    const item = @workItemId(0);
    const count = 16;
    const block: [*]u32 = @ptrCast(@alignCast(nvptx.malloc(count * @sizeOf(u32)) orelse {
        fail(out, 1);
        return;
    }));
    for (0..count) |i| block[i] = item * 1000 + @as(u32, @intCast(i));
    shared_pointers[item] = &block[0];
    nvptx.barrierSyncAligned(.work_group, null);
    for (0..count) |i| expect(out, 2, block[i] == item * 1000 + @as(u32, @intCast(i)));
    const neighbor = shared_pointers[(item + 1) % 128].?;
    expect(out, 3, @intFromPtr(neighbor) != @intFromPtr(&block[0]));
    nvptx.barrierSyncAligned(.work_group, null);
    nvptx.free(block);
    const again: [*]u32 = @ptrCast(@alignCast(nvptx.malloc(count * @sizeOf(u32)) orelse {
        fail(out, 4);
        return;
    }));
    again[0] = 0xdeadbeef;
    expect(out, 5, again[0] == 0xdeadbeef);
    nvptx.free(again);
}
