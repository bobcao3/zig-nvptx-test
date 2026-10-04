const std = @import("std");
const nvptx = std.nvptx;

const Barrier = enum(u4) { work_group = 0, pair = 1 };

var exchange: [64]u32 addrspace(.shared) = undefined;

fn within(value: f32, expected: f32, tolerance: f32) u32 {
    return @intFromBool(@abs(value - expected) <= tolerance);
}

fn shuffleDownExpected(lane: u32, delta: u32, width: u32) u32 {
    const source = lane + delta;
    return if (source / width == lane / width) source else lane;
}

/// Launch with 2 work groups of 64 work items. Work item 0 of work group 0 writes the results.
export fn stdNvptx(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const tid = @workItemId(0);
    const lane = nvptx.laneid();
    var words: [32]u32 = @splat(0);

    words[0] = nvptx.voteBallotSync(lane % 3 == 0, .all);
    words[1] = @intFromBool(nvptx.voteSync(.all, lane < 32, .all));
    words[2] = @intFromBool(nvptx.voteSync(.all, lane < 31, .all));
    words[3] = @intFromBool(nvptx.voteSync(.any, lane == 7, .all));
    words[4] = @intFromBool(nvptx.voteSync(.uni, lane < 16, .all));
    words[5] = @intFromBool(nvptx.voteSync(.uni, tid < 64, .all));
    words[6] = nvptx.voteBallotSync(lane == tid % 32, .all);

    const down = nvptx.shflSync(.down, u32, lane, 3, .fromWidth(.down, 16), .all);
    words[7] = nvptx.voteBallotSync(down == shuffleDownExpected(lane, 3, 16), .all);
    const up = nvptx.shflSync(.up, u32, lane * 10, 1, .fromWidth(.up, 32), .all);
    words[8] = nvptx.voteBallotSync(up == if (lane == 0) 0 else (lane - 1) * 10, .all);
    const bfly = nvptx.shflSync(.bfly, f32, @floatFromInt(lane), 1, .fromWidth(.bfly, 32), .all);
    words[9] = nvptx.voteBallotSync(bfly == @as(f32, @floatFromInt(lane ^ 1)), .all);
    words[10] = nvptx.shflSync(.idx, u32, lane * 10, 5, .fromWidth(.idx, 32), .all);
    const idx8 = nvptx.shflSync(.idx, i32, @intCast(lane), 2, .fromWidth(.idx, 8), .all);
    words[11] = nvptx.voteBallotSync(idx8 == (lane & ~@as(u32, 7)) + 2, .all);

    words[12] = nvptx.reduxSync(.add, u32, lane, .all);
    words[13] = @bitCast(nvptx.reduxSync(.max, i32, @as(i32, @intCast(lane)) - 20, .all));
    words[14] = @bitCast(nvptx.reduxSync(.min, i32, @as(i32, @intCast(lane)) - 20, .all));
    words[15] = nvptx.reduxSync(.@"or", u32, @as(u32, 1) << @intCast(lane % 8), .all);
    words[16] = nvptx.reduxSync(.xor, u32, lane, .all);

    words[17] = nvptx.activemask();
    if (lane < 10) words[18] = nvptx.activemask();
    words[19] = nvptx.nctaid(0);
    words[20] = nvptx.nctaid(1);
    const start = nvptx.clock64();
    const start32 = nvptx.clock();
    nvptx.fence(.seq_cst, .gpu);
    nvptx.fence(.acq_rel, .cta);
    nvptx.fence(.seq_cst, .sys);
    words[21] = @intFromBool(nvptx.clock64() > start and nvptx.clock() != start32);

    exchange[tid] = tid * 2;
    nvptx.barrierSyncAligned(.work_group, null);
    words[22] = nvptx.voteBallotSync(exchange[63 - tid] == (63 - tid) * 2, .all);
    nvptx.barrierSyncAligned(Barrier.pair, 64);
    exchange[tid] = tid * 3;
    nvptx.barrierSyncAligned(Barrier.pair, 64);
    words[23] = nvptx.voteBallotSync(exchange[63 - tid] == (63 - tid) * 3, .all);
    nvptx.barrierSyncAligned(.work_group, null);
    exchange[tid] = tid + 100;
    nvptx.barWarpSync(.all);
    words[24] = nvptx.voteBallotSync(exchange[tid ^ 31] == (tid ^ 31) + 100, .all);

    words[25] = within(nvptx.ex2Approx(3.0), 8.0, 2 * std.math.floatEps(f32) * 8.0);
    words[26] = within(nvptx.lg2Approx(8.0), 3.0, 3.0 * 0x1p-22);
    words[27] = within(nvptx.sinApprox(0.5), 0.47942554, 0x1p-20);
    words[28] = within(nvptx.cosApprox(0.5), 0.87758255, 0x1p-20);
    words[29] = within(nvptx.rcpApprox(3.0), 1.0 / 3.0, std.math.floatEps(f32));
    words[30] = within(nvptx.rsqrtApprox(4.0), 0.5, 0.5 * 0x1p-22);
    words[31] = within(nvptx.divApprox(1.0, 3.0), 1.0 / 3.0, std.math.floatEps(f32));

    if (@workGroupId(0) == 0 and tid == 0) {
        for (words, 0..) |word, i| out[i] = word;
    }
}
