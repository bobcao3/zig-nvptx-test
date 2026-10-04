const nv = @import("std").nvptx;
const std = @import("std");

const Out = [*]addrspace(.global) u32;

var shared_words: [8]u32 addrspace(.shared) = undefined;
var shared_wide: u64 addrspace(.shared) = undefined;
var shared_histogram: [16]u32 addrspace(.shared) = undefined;
var shared_flags: [128]u32 addrspace(.shared) = undefined;

fn tid() u32 {
    return @workItemId(0);
}

fn group() void {
    nv.barrierSyncAligned(.work_group, null);
}

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

fn at(comptime T: type, out: Out, word: usize) *addrspace(.global) T {
    return @ptrCast(@alignCast(out + word));
}

const scopes = [_]nv.Scope{ .cta, .gpu, .sys };
const orders = [_]std.lang.AtomicOrder{ .monotonic, .acquire, .release, .acq_rel };

export fn atom_add_scopes(out: Out) callconv(.kernel) void {
    const n: u32 = @workGroupSize(0);
    inline for (scopes, 0..) |scope, s| {
        inline for (orders, 0..) |order, o| {
            const slot = 16 + 2 * (s * orders.len + o);
            const old = nv.atom(scope, order, .add, &out[slot], 1);
            _ = nv.atom(scope, order, .add, &out[slot + 1], old);
        }
    }
    group();
    if (tid() == 0) {
        for (0..scopes.len * orders.len) |i| {
            expect(out, @src().line, out[16 + 2 * i] == n);
            expect(out, @src().line, out[17 + 2 * i] == n * (n - 1) / 2);
        }
    }
}

export fn atom_add_shared(out: Out) callconv(.kernel) void {
    const n: u32 = @workGroupSize(0);
    if (tid() == 0) shared_words = @splat(0);
    group();
    const old = nv.atom(.cta, .monotonic, .add, &shared_words[0], 1);
    _ = nv.atom(.gpu, .acq_rel, .add, &shared_words[1], old);
    group();
    if (tid() == 0) {
        expect(out, @src().line, shared_words[0] == n);
        expect(out, @src().line, shared_words[1] == n * (n - 1) / 2);
    }
}

export fn atom_integer_ops(out: Out) callconv(.kernel) void {
    const t = tid();
    if (t == 0) {
        out[16] = 0xffff_ffff;
        out[17] = 0;
        out[18] = 0;
        out[19] = 0xffff_ffff;
        out[20] = 0x8000_0000;
        out[21] = 0;
    }
    group();
    _ = nv.atom(.gpu, .monotonic, .min, &out[16], t + 10);
    _ = nv.atom(.cta, .acq_rel, .max, &out[17], t + 10);
    _ = nv.atom(.sys, .release, .@"or", &out[18], @as(u32, 1) << @intCast(t % 32));
    _ = nv.atom(.gpu, .acquire, .@"and", &out[19], ~(@as(u32, 1) << @intCast(t % 32)));
    _ = nv.atom(.gpu, .monotonic, .max, at(i32, out, 20), -@as(i32, @intCast(t)));
    _ = nv.atom(.cta, .monotonic, .xor, &out[21], t + 1);
    group();
    if (t == 0) {
        expect(out, @src().line, out[16] == 10);
        expect(out, @src().line, out[17] == 10 + @workGroupSize(0) - 1);
        expect(out, @src().line, out[18] == 0xffff_ffff);
        expect(out, @src().line, out[19] == 0);
        expect(out, @src().line, at(i32, out, 20).* == 0);
        var expected_xor: u32 = 0;
        for (0..@workGroupSize(0)) |i| expected_xor ^= @as(u32, @intCast(i)) + 1;
        expect(out, @src().line, out[21] == expected_xor);
    }
}

export fn atom_signed_and_wide(out: Out) callconv(.kernel) void {
    const t = tid();
    if (t == 0) {
        at(i32, out, 16).* = 100;
        at(i64, out, 18).* = 100;
        at(u64, out, 20).* = 0;
        at(u64, out, 22).* = 0;
        at(f32, out, 24).* = 0;
        at(f64, out, 26).* = 0;
    }
    group();
    _ = nv.atom(.gpu, .monotonic, .min, at(i32, out, 16), 50 - @as(i32, @intCast(t)));
    _ = nv.atom(.gpu, .monotonic, .min, at(i64, out, 18), -@as(i64, @intCast(t)) * 0x1_0000_0000);
    _ = nv.atom(.gpu, .acq_rel, .add, at(u64, out, 20), 0x1_0000_0001);
    _ = nv.atom(.cta, .monotonic, .max, at(u64, out, 22), @as(u64, t) << 32);
    _ = nv.atom(.gpu, .monotonic, .add, at(f32, out, 24), 0.5);
    _ = nv.atom(.sys, .monotonic, .add, at(f64, out, 26), 0.25);
    group();
    if (t == 0) {
        const n = @workGroupSize(0);
        expect(out, @src().line, at(i32, out, 16).* == 50 - @as(i32, @intCast(n - 1)));
        expect(out, @src().line, at(i64, out, 18).* == -@as(i64, @intCast(n - 1)) * 0x1_0000_0000);
        expect(out, @src().line, at(u64, out, 20).* == @as(u64, n) * 0x1_0000_0001);
        expect(out, @src().line, at(u64, out, 22).* == @as(u64, n - 1) << 32);
        expect(out, @src().line, at(f32, out, 24).* == @as(f32, @floatFromInt(n)) * 0.5);
        expect(out, @src().line, at(f64, out, 26).* == @as(f64, @floatFromInt(n)) * 0.25);
    }
}

export fn atom_exchange_and_compare(out: Out) callconv(.kernel) void {
    const t = tid();
    if (t == 0) {
        out[16] = 7;
        out[17] = 0;
        at(u64, out, 18).* = 0;
        at(f32, out, 20).* = 1.5;
    }
    group();
    const previous = nv.atom(.gpu, .acq_rel, .exch, &out[16], 1000 + t);
    _ = nv.atom(.gpu, .monotonic, .add, &out[17], previous);
    var expected: u64 = nv.atomCas(.gpu, .acquire, &out[18], 0, 0);
    while (true) {
        const seen = nv.atomCas(.cta, .acq_rel, at(u64, out, 18), expected, expected + (t + 1));
        if (seen == expected) break;
        expected = seen;
    }
    if (t == 0) expect(out, @src().line, nv.atom(.gpu, .monotonic, .exch, at(f32, out, 20), 2.5) == 1.5);
    group();
    if (t == 0) {
        const n: u32 = @workGroupSize(0);
        expect(out, @src().line, out[16] >= 1000 and out[16] < 1000 + n);
        var expected_sum: u32 = 7;
        for (0..n) |i| expected_sum +%= 1000 + @as(u32, @intCast(i));
        expect(out, @src().line, out[17] == expected_sum -% out[16]);
        expect(out, @src().line, at(u64, out, 18).* == @as(u64, n) * (n + 1) / 2);
        expect(out, @src().line, at(f32, out, 20).* == 2.5);
    }
}

export fn atom_inc_dec(out: Out) callconv(.kernel) void {
    const t = tid();
    if (t == 0) {
        shared_histogram = @splat(0);
        shared_words = @splat(0);
        shared_words[1] = 9;
        out[16] = 15;
        out[17] = 0;
        out[18] = 50;
        out[19] = 9;
        out[20] = 5;
        out[21] = 8;
        expect(out, @src().line, nv.atom(.gpu, .monotonic, .inc, &out[16], 9) == 15);
        expect(out, @src().line, nv.atom(.gpu, .monotonic, .dec, &out[17], 9) == 0);
        expect(out, @src().line, nv.atom(.gpu, .monotonic, .dec, &out[18], 9) == 50);
        expect(out, @src().line, nv.atom(.cta, .monotonic, .inc, &out[19], 9) == 9);
        expect(out, @src().line, nv.atom(.cta, .monotonic, .inc, &out[20], 5) == 5);
        expect(out, @src().line, nv.atom(.cta, .monotonic, .dec, &out[21], 8) == 8);
        expect(out, @src().line, out[16] == 0);
        expect(out, @src().line, out[17] == 9);
        expect(out, @src().line, out[18] == 9);
        expect(out, @src().line, out[19] == 0);
        expect(out, @src().line, out[20] == 0);
        expect(out, @src().line, out[21] == 7);
    }
    group();
    const increased = nv.atom(.cta, .monotonic, .inc, &shared_words[0], 9);
    _ = nv.atom(.cta, .monotonic, .add, &shared_histogram[increased], 1);
    const decreased = nv.atom(.gpu, .acq_rel, .dec, &shared_words[1], 9);
    _ = nv.atom(.cta, .monotonic, .add, &shared_histogram[decreased + 0], 0x10000);
    group();
    if (t == 0) {
        const n: u32 = @workGroupSize(0);
        expect(out, @src().line, shared_words[0] == n % 10);
        expect(out, @src().line, shared_words[1] == 9 - n % 10);
        for (0..10) |i| {
            const rounds: u32 = n / 10 + @intFromBool(i < n % 10);
            const decrement_rounds: u32 = n / 10 + @intFromBool(i > 9 - n % 10);
            expect(out, @src().line, shared_histogram[i] & 0xffff == rounds);
            expect(out, @src().line, shared_histogram[i] >> 16 == decrement_rounds);
        }
    }
}

export fn red_operations(out: Out) callconv(.kernel) void {
    const t = tid();
    if (t == 0) {
        out[16] = 0;
        out[17] = 0xffff_ffff;
        out[18] = 0;
        at(u64, out, 20).* = 0;
        out[22] = 0;
        out[23] = 0;
        out[24] = 0;
        shared_wide = 0;
    }
    group();
    nv.red(.gpu, .monotonic, .add, &out[16], 3);
    nv.red(.cta, .release, .min, &out[17], t);
    nv.red(.sys, .monotonic, .max, &out[18], t);
    nv.red(.gpu, .release, .add, at(u64, out, 20), 0x1_0000_0000);
    nv.red(.cta, .monotonic, .inc, &out[22], 15);
    nv.red(.gpu, .monotonic, .xor, &out[23], t);
    nv.red(.cta, .monotonic, .add, &shared_wide, 2);
    nv.red(.gpu, .monotonic, .@"or", &out[24], @as(u32, 1) << @intCast(t % 32));
    group();
    if (t == 0) {
        const n: u32 = @workGroupSize(0);
        expect(out, @src().line, out[16] == 3 * n);
        expect(out, @src().line, out[17] == 0);
        expect(out, @src().line, out[18] == n - 1);
        expect(out, @src().line, at(u64, out, 20).* == @as(u64, n) << 32);
        expect(out, @src().line, out[22] == n % 16);
        var expected_xor: u32 = 0;
        for (0..n) |i| expected_xor ^= @as(u32, @intCast(i));
        expect(out, @src().line, out[23] == expected_xor);
        expect(out, @src().line, shared_wide == 2 * n);
        expect(out, @src().line, out[24] == 0xffff_ffff);
    }
}

export fn atom_add_half(out: Out) callconv(.kernel) void {
    const t = tid();
    const Half2 = @Vector(2, f16);
    if (t == 0) {
        at(f16, out, 16).* = 0;
        at(Half2, out, 17).* = .{ 0, 0 };
        at(f16, out, 20).* = 0;
        at(Half2, out, 21).* = .{ 0, 0 };
    }
    group();
    _ = nv.atom(.gpu, .monotonic, .add, at(f16, out, 16), 1);
    _ = nv.atom(.cta, .acq_rel, .add, at(Half2, out, 17), .{ 1, 2 });
    nv.red(.gpu, .monotonic, .add, at(f16, out, 20), 2);
    nv.red(.sys, .release, .add, at(Half2, out, 21), .{ 0.5, 4 });
    group();
    if (t == 0) {
        const n: f16 = @floatFromInt(@workGroupSize(0));
        expect(out, @src().line, at(f16, out, 16).* == n);
        expect(out, @src().line, at(Half2, out, 17).*[0] == n);
        expect(out, @src().line, at(Half2, out, 17).*[1] == 2 * n);
        expect(out, @src().line, at(f16, out, 20).* == 2 * n);
        expect(out, @src().line, at(Half2, out, 21).*[0] == n / 2);
        expect(out, @src().line, at(Half2, out, 21).*[1] == 4 * n);
    }
}

export fn match_any_groups(out: Out) callconv(.kernel) void {
    const lane = nv.laneid();
    const mask32 = nv.matchAnySync(u32, lane % 3, .all);
    var expected32: u32 = 0;
    for (0..32) |i| {
        if (i % 3 == lane % 3) expected32 |= @as(u32, 1) << @intCast(i);
    }
    expect(out, @src().line, mask32 == expected32);

    const wide = @as(u64, lane % 5) << 33 | 1;
    const mask64 = nv.matchAnySync(u64, wide, .all);
    var expected64: u32 = 0;
    for (0..32) |i| {
        if (i % 5 == lane % 5) expected64 |= @as(u32, 1) << @intCast(i);
    }
    expect(out, @src().line, mask64 == expected64);

    const unique = nv.matchAnySync(f32, @floatFromInt(lane), .all);
    expect(out, @src().line, unique == @as(u32, 1) << @intCast(lane));

    if (lane < 16) expect(out, @src().line, nv.matchAnySync(u32, 0, @fromBackingInt(0x0000_ffff)) == 0xffff);
}

export fn match_all_groups(out: Out) callconv(.kernel) void {
    const lane = nv.laneid();
    const same = nv.matchAllSync(u32, 7, .all);
    expect(out, @src().line, same.mask == 0xffff_ffff and same.all_equal);
    const differ = nv.matchAllSync(u32, if (lane == 31) 8 else 7, .all);
    expect(out, @src().line, differ.mask == 0 and !differ.all_equal);
    const wide = nv.matchAllSync(u64, @as(u64, 0x1_0000_0000) + (lane / 32), .all);
    expect(out, @src().line, wide.mask == 0xffff_ffff and wide.all_equal);
    if (lane < 16) {
        const half = nv.matchAllSync(u32, lane / 16, @fromBackingInt(0x0000_ffff));
        expect(out, @src().line, half.mask == 0xffff and half.all_equal);
    }
}

export fn elect_sync(out: Out) callconv(.kernel) void {
    if (comptime nv.gpu.sm < 90 or nv.gpu.ptx < 80) return;
    const leader = nv.voteBallotSync(nv.electSync(.all), .all);
    expect(out, @src().line, @popCount(leader) == 1);
    expect(out, @src().line, nv.voteBallotSync(nv.electSync(.all), .all) == leader);
    if (nv.laneid() % 2 == 1) {
        const odd: nv.MemberMask = @fromBackingInt(0xaaaa_aaaa);
        const odd_leader = nv.voteBallotSync(nv.electSync(odd), odd);
        expect(out, @src().line, @popCount(odd_leader) == 1 and odd_leader & 0x5555_5555 == 0);
    }
}

export fn barrier_reductions(out: Out) callconv(.kernel) void {
    const t = tid();
    const n: u32 = @workGroupSize(0);
    expect(out, @src().line, nv.barrierRedPopcAligned(.work_group, null, t % 3 == 0) == (n + 2) / 3);
    expect(out, @src().line, nv.barrierRedPopcAligned(.work_group, null, false) == 0);
    expect(out, @src().line, nv.barrierRedPopcAligned(.work_group, 128, true) == 128);
    expect(out, @src().line, nv.barrierRedAligned(.@"and", .work_group, null, true));
    expect(out, @src().line, !nv.barrierRedAligned(.@"and", .work_group, null, t != 5));
    expect(out, @src().line, nv.barrierRedAligned(.@"or", .work_group, null, t == 77));
    expect(out, @src().line, !nv.barrierRedAligned(.@"or", .work_group, 128, false));
}

export fn barrier_arrive(out: Out) callconv(.kernel) void {
    const t = tid();
    const Stage = enum(u4) { ready = 1, _ };
    if (t < 64) {
        shared_flags[t] = t + 1;
        nv.barrierArriveAligned(Stage.ready, 128);
    } else {
        nv.barrierSyncAligned(Stage.ready, 128);
        expect(out, @src().line, shared_flags[t - 64] == t - 63);
    }
    group();
}

export fn grid_barrier(out: Out) callconv(.kernel) void {
    const barrier: nv.GridBarrier = .{ .counter = &out[1] };
    const groups = nv.nctaid(0);
    const me = @workGroupId(0);
    const slots = out + 16;
    const rounds = 5;
    for (1..rounds + 1) |round| {
        if (tid() == 0) slots[me] = @intCast(round * 1000 + me);
        barrier.sync();
        for (0..groups) |g| expect(out, @src().line, slots[g] == @as(u32, @intCast(round * 1000 + g)));
        barrier.sync();
    }
}
