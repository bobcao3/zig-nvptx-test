const std = @import("std");
const nv = std.nvptx;
const cache = nv.cache;

const Out = [*]addrspace(.global) u32;
const Mbarrier = nv.Mbarrier;

const work_items = 128;

const has_ptx71 = nv.gpu.ptx >= 71;
const has_ptx74 = nv.gpu.ptx >= 74;
const has_try_wait = nv.gpu.sm >= 90 and nv.gpu.ptx >= 78;
const has_transactions = nv.gpu.sm >= 90 and nv.gpu.ptx >= 80;

var shared_words: [work_items]u32 align(16) addrspace(.shared) = undefined;
var shared_wide: [work_items]u64 align(16) addrspace(.shared) = undefined;
var shared_quads: [work_items][4]u32 align(16) addrspace(.shared) = undefined;
var shared_stages: [2][work_items]u32 align(16) addrspace(.shared) = undefined;
var barrier: Mbarrier addrspace(.shared) = undefined;
var barrier_other: Mbarrier addrspace(.shared) = undefined;
var shared_flag: u32 addrspace(.shared) = undefined;

fn tid() u32 {
    return @workItemId(0);
}

fn group() void {
    nv.barrierSyncAligned(.work_group, null);
}

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

fn view(comptime T: type, out: Out, word: u32) *addrspace(.global) T {
    return @ptrCast(@alignCast(out + word));
}

fn pattern(word: u32) u32 {
    return word *% 2654435761 +% 7;
}

fn bytesOf(value: anytype) [@sizeOf(@TypeOf(value))]u8 {
    return @bitCast(value);
}

fn sameBytes(a: anytype, b: @TypeOf(a)) bool {
    const left = bytesOf(a);
    const right = bytesOf(b);
    var equal = true;
    inline for (left, right) |x, y| equal = equal and x == y;
    return equal;
}

fn imageOf(comptime T: type, words: [4]u32) T {
    const bytes: [16]u8 = @bitCast(words);
    return @bitCast(bytes[0..@sizeOf(T)].*);
}

fn same(a: [4]u32, b: [4]u32) bool {
    return @reduce(.And, @as(@Vector(4, u32), a) == @as(@Vector(4, u32), b));
}

fn spin(m: *addrspace(.shared) Mbarrier, token: Mbarrier.ArrivalToken) void {
    while (!m.testWait(token)) {}
}

const data_word = 256;

fn fillData(out: Out) [4]u32 {
    const first = data_word + 4 * tid();
    var words: [4]u32 = undefined;
    for (&words, 0..) |*word, i| {
        word.* = pattern(first + @as(u32, @intCast(i)));
        out[first + i] = word.*;
    }
    return words;
}

export fn cp_async_sizes(out: Out) callconv(.kernel) void {
    const i = tid();
    const words = fillData(out);
    group();
    const source: Out = out + data_word + 4 * i;

    shared_words[i] = 0;
    shared_wide[i] = 0;
    shared_quads[i] = @splat(0);
    nv.cp_async.copy(.cache_all_levels, &shared_words[i], &source[0]);
    nv.cp_async.copy(.cache_all_levels, &shared_wide[i], view(u64, out, data_word + 4 * i));
    nv.cp_async.copy(.cache_all_levels, &shared_quads[i], view([4]u32, out, data_word + 4 * i));
    nv.cp_async.commitGroup();
    nv.cp_async.waitAll();
    expect(out, @src().line, shared_words[i] == words[0]);
    expect(out, @src().line, shared_wide[i] == @as(u64, words[1]) << 32 | words[0]);
    expect(out, @src().line, same(shared_quads[i], words));

    shared_quads[i] = @splat(0);
    nv.cp_async.copy(.cache_global_level, &shared_quads[i], view([4]u32, out, data_word + 4 * i));
    nv.cp_async.commitGroup();
    nv.cp_async.waitGroup(0);
    expect(out, @src().line, same(shared_quads[i], words));
}

export fn cp_async_zero_fill(out: Out) callconv(.kernel) void {
    const i = tid();
    const words = fillData(out);
    group();
    const src_size: u32 = (i % 5) * 4;
    shared_quads[i] = @splat(0xaaaaaaaa);
    nv.cp_async.copyZeroFill(.cache_global_level, &shared_quads[i], view([4]u32, out, data_word + 4 * i), src_size);
    nv.cp_async.commitGroup();
    nv.cp_async.waitAll();
    const quad: [4]u32 = shared_quads[i];
    for (quad, words, 0..) |got, word, lane| {
        const expected: u32 = if (lane * 4 < src_size) word else 0;
        expect(out, @src().line, got == expected);
    }

    shared_words[i] = 0xaaaaaaaa;
    nv.cp_async.copyZeroFill(.cache_all_levels, &shared_words[i], &out[data_word + 4 * i], 0);
    nv.cp_async.waitAll();
    expect(out, @src().line, shared_words[i] == 0);

    shared_wide[i] = ~@as(u64, 0);
    nv.cp_async.copyZeroFill(.cache_all_levels, &shared_wide[i], view(u64, out, data_word + 4 * i), 3);
    nv.cp_async.waitAll();
    expect(out, @src().line, shared_wide[i] == @as(u64, words[0]) & 0xffffff);
}

export fn cp_async_policy(out: Out) callconv(.kernel) void {
    if (comptime !has_ptx74) return;
    const i = tid();
    const words = fillData(out);
    group();
    const policy = cache.createpolicy(.evict_last, .evict_unchanged, 0.5);
    shared_quads[i] = @splat(0);
    nv.cp_async.copyWithPolicy(.cache_global_level, &shared_quads[i], view([4]u32, out, data_word + 4 * i), policy);
    shared_words[i] = 0;
    nv.cp_async.copyWithPolicy(.cache_all_levels, &shared_words[i], &out[data_word + 4 * i + 1], policy);
    nv.cp_async.commitGroup();
    nv.cp_async.waitGroup(0);
    expect(out, @src().line, same(shared_quads[i], words));
    expect(out, @src().line, shared_words[i] == words[1]);
}

/// Copies 4 tiles through 2 stages. The work item i adds the word that work item i + 1 copied.
export fn cp_async_pipeline(out: Out) callconv(.kernel) void {
    const tiles = 4;
    const i = tid();
    for (0..tiles) |tile| out[512 + tile * work_items + i] = pattern(@intCast(tile * work_items + i));
    group();

    var sum: u32 = 0;
    nv.cp_async.copy(.cache_all_levels, &shared_stages[0][i], &out[512 + i]);
    nv.cp_async.commitGroup();
    for (0..tiles) |tile| {
        if (tile + 1 < tiles) {
            nv.cp_async.copy(
                .cache_all_levels,
                &shared_stages[(tile + 1) % 2][i],
                &out[512 + (tile + 1) * work_items + i],
            );
            nv.cp_async.commitGroup();
            nv.cp_async.waitGroup(1);
        } else {
            nv.cp_async.waitGroup(0);
        }
        group();
        sum +%= shared_stages[tile % 2][(i + 1) % work_items];
        group();
    }

    var expected: u32 = 0;
    for (0..tiles) |tile| expected +%= pattern(@intCast(tile * work_items + (i + 1) % work_items));
    expect(out, @src().line, sum == expected);
}

export fn cp_async_mbarrier(out: Out) callconv(.kernel) void {
    const i = tid();
    const words = fillData(out);
    if (i == 0) {
        barrier.init(work_items);
        barrier_other.init(2 * work_items);
    }
    group();

    shared_words[i] = 0;
    nv.cp_async.copy(.cache_all_levels, &shared_words[i], &out[data_word + 4 * i]);
    nv.cp_async.mbarrierArrive(&barrier);
    const token = barrier.arrive();
    spin(&barrier, token);
    expect(out, @src().line, shared_words[i] == words[0]);
    group();

    shared_wide[i] = 0;
    nv.cp_async.copy(.cache_all_levels, &shared_wide[i], view(u64, out, data_word + 4 * i));
    nv.cp_async.mbarrierArriveNoinc(&barrier_other);
    const token_other = barrier_other.arrive();
    spin(&barrier_other, token_other);
    expect(out, @src().line, shared_wide[i] == @as(u64, words[1]) << 32 | words[0]);
    group();
    if (i == 0) {
        barrier.inval();
        barrier_other.inval();
    }
}

fn recordRound(round: u32, who: u32) u32 {
    return round * 1000 + who;
}

export fn mbarrier_phases(out: Out) callconv(.kernel) void {
    const i = tid();
    if (i == 0) barrier.init(work_items);
    group();
    for (0..6) |round| {
        const r: u32 = @intCast(round);
        shared_stages[round % 2][i] = recordRound(r, i);
        const token = barrier.arrive();
        spin(&barrier, token);
        const neighbor = (i + 1) % work_items;
        expect(out, @src().line, shared_stages[round % 2][neighbor] == recordRound(r, neighbor));
    }
    group();
    if (i == 0) barrier.inval();
}

export fn mbarrier_parity(out: Out) callconv(.kernel) void {
    if (comptime !has_ptx71) return;
    const i = tid();
    if (i == 0) barrier.init(work_items);
    group();
    for (0..6) |round| {
        const r: u32 = @intCast(round);
        shared_stages[round % 2][i] = recordRound(r, i);
        _ = barrier.arrive();
        while (!barrier.testWaitParity(@intCast(round % 2))) {}
        const neighbor = (i + 1) % work_items;
        expect(out, @src().line, shared_stages[round % 2][neighbor] == recordRound(r, neighbor));
    }
    group();
    if (i == 0) barrier.inval();
}

export fn mbarrier_counts(out: Out) callconv(.kernel) void {
    if (comptime !has_ptx71) return;
    if (tid() == 0) {
        barrier.init(10);
        const first = barrier.arriveNoComplete(3);
        expect(out, @src().line, first.pendingCount() == 10);
        const second = barrier.arriveNoComplete(2);
        expect(out, @src().line, second.pendingCount() == 7);
        expect(out, @src().line, !barrier.testWait(first));
        var last = second;
        for (0..5) |_| last = barrier.arrive();
        expect(out, @src().line, barrier.testWait(first));
        expect(out, @src().line, barrier.testWait(last));
        expect(out, @src().line, barrier.testWaitParity(0));
        expect(out, @src().line, !barrier.testWaitParity(1));
        barrier.inval();

        barrier_other.init(10);
        const dropped = barrier_other.arriveDropNoComplete(2);
        expect(out, @src().line, dropped.pendingCount() == 10);
        for (0..8) |_| _ = barrier_other.arrive();
        var phase_one: Mbarrier.ArrivalToken = undefined;
        for (0..8) |_| phase_one = barrier_other.arrive();
        expect(out, @src().line, barrier_other.testWait(phase_one));
        barrier_other.inval();

        barrier.init(1);
        spin(&barrier, barrier.arrive());
        barrier.inval();
    }
}

export fn mbarrier_drop(out: Out) callconv(.kernel) void {
    const i = tid();
    if (i == 0) barrier.init(work_items);
    group();
    if (i >= work_items / 2) {
        _ = barrier.arriveDrop();
        return;
    }
    spin(&barrier, barrier.arrive());
    const token = barrier.arrive();
    spin(&barrier, token);
    expect(out, @src().line, barrier.testWait(token));
}

export fn mbarrier_wait_parity(out: Out) callconv(.kernel) void {
    if (comptime !has_try_wait) return;
    const i = tid();
    if (i == 0) barrier.init(work_items);
    group();
    for (0..6) |round| {
        const r: u32 = @intCast(round);
        shared_stages[round % 2][i] = recordRound(r, i);
        _ = barrier.arrive();
        while (!barrier.tryWaitParity(@intCast(round % 2))) {}
        const neighbor = (i + 1) % work_items;
        expect(out, @src().line, shared_stages[round % 2][neighbor] == recordRound(r, neighbor));
    }
    group();
    if (i == 0) barrier.inval();
}

export fn mbarrier_transactions(out: Out) callconv(.kernel) void {
    if (comptime !has_transactions) return;
    const i = tid();
    if (i == 0) {
        barrier.init(1);
        _ = barrier.arriveExpectTx(64);
        expect(out, @src().line, !barrier.testWaitParity(0));
        barrier.completeTx(48);
        expect(out, @src().line, !barrier.testWaitParity(0));
        barrier.completeTx(16);
        expect(out, @src().line, barrier.tryWaitParity(0));
        expect(out, @src().line, !barrier.testWaitParity(1));
        barrier.inval();

        barrier.init(1);
        barrier.completeTx(8);
        expect(out, @src().line, !barrier.testWaitParity(0));
        _ = barrier.arriveExpectTx(8);
        expect(out, @src().line, barrier.tryWaitParity(0));
        barrier.inval();

        barrier.init(1);
        _ = barrier.arriveExpectTx(4 * work_items);
    }
    group();
    barrier.completeTx(4);
    while (!barrier.tryWaitParity(0)) {}
    group();
    if (i == 0) barrier.inval();
}

const load_operators = [_]cache.LoadOperator{
    .cache_all_levels,
    .cache_global_level,
    .cache_streaming,
    .last_use,
    .cache_volatile,
};

const l1_priorities = [_]cache.L1EvictionPriority{
    .evict_normal,
    .evict_unchanged,
    .evict_first,
    .evict_last,
    .no_allocate,
};

const non_coherent_operators = [_]cache.NonCoherentOperator{
    .cache_all_levels,
    .cache_global_level,
    .cache_streaming,
};

const store_operators = [_]cache.StoreOperator{
    .write_back,
    .cache_global_level,
    .cache_streaming,
    .write_through,
};

const Quad = @Vector(4, u32);
const element_types = .{
    u8,
    u16,
    u32,
    u64,
    f32,
    f64,
    @Vector(2, u8),
    @Vector(4, u8),
    @Vector(2, u16),
    @Vector(4, u16),
    @Vector(2, u32),
    Quad,
    @Vector(2, f32),
    @Vector(4, f32),
    @Vector(2, u64),
    @Vector(2, f64),
};

fn loadsMatch(comptime load: anytype, out: Out, words: [4]u32) bool {
    var ok = true;
    inline for (element_types) |T| {
        const got = load(T, view(T, out, data_word + 4 * tid()));
        ok = ok and sameBytes(got, imageOf(T, words));
    }
    return ok;
}

fn LoadWith(comptime operator: anytype) type {
    return struct {
        fn call(comptime T: type, ptr: *addrspace(.global) T) T {
            return switch (@TypeOf(operator)) {
                cache.LoadOperator => cache.load(.{ .operator = operator }, ptr),
                cache.L1EvictionPriority => cache.load(.{ .l1_eviction_priority = operator }, ptr),
                cache.NonCoherentOperator => cache.loadNonCoherent(operator, ptr),
                else => unreachable,
            };
        }
    };
}

export fn cache_load_operators(out: Out) callconv(.kernel) void {
    const words = fillData(out);
    group();
    inline for (load_operators) |operator| {
        expect(out, @src().line, loadsMatch(LoadWith(operator).call, out, words));
    }
    if (comptime has_ptx74) {
        inline for (l1_priorities) |priority| {
            expect(out, @src().line, loadsMatch(LoadWith(priority).call, out, words));
        }
    }
    inline for (non_coherent_operators) |operator| {
        expect(out, @src().line, loadsMatch(LoadWith(operator).call, out, words));
    }
}

export fn cache_store_operators(out: Out) callconv(.kernel) void {
    const first = data_word + 4 * tid();
    const words: [4]u32 = .{ pattern(first), pattern(first + 1), pattern(first + 2), pattern(first + 3) };
    inline for (store_operators) |operator| {
        inline for (element_types) |T| {
            view([4]u32, out, first).* = @splat(0);
            cache.store(.{ .operator = operator }, view(T, out, first), imageOf(T, words));
            const bytes: [16]u8 = @bitCast(view([4]u32, out, first).*);
            const expected: [16]u8 = padded(T, words);
            expect(out, @src().line, sameBytes(bytes, expected));
        }
    }
    if (comptime has_ptx74) {
        inline for (l1_priorities) |priority| {
            inline for (element_types) |T| {
                view([4]u32, out, first).* = @splat(0);
                cache.store(.{ .l1_eviction_priority = priority }, view(T, out, first), imageOf(T, words));
                const bytes: [16]u8 = @bitCast(view([4]u32, out, first).*);
                const expected: [16]u8 = padded(T, words);
                expect(out, @src().line, sameBytes(bytes, expected));
            }
        }
    }
}

fn padded(comptime T: type, words: [4]u32) [16]u8 {
    var bytes: [16]u8 = @splat(0);
    const image: [@sizeOf(T)]u8 = @bitCast(imageOf(T, words));
    bytes[0..@sizeOf(T)].* = image;
    return bytes;
}

const primaries = [_]cache.L2EvictionPriority{ .evict_last, .evict_normal, .evict_first, .evict_unchanged };
const secondaries = [_]?cache.L2EvictionPriority{ null, .evict_first, .evict_unchanged };

export fn cache_policies(out: Out) callconv(.kernel) void {
    if (comptime !has_ptx74) return;
    const first = data_word + 4 * tid();
    const words = fillData(out);
    group();
    inline for (primaries) |primary| {
        inline for (secondaries) |secondary| {
            inline for (.{ 1.0, 0.5, 0.0625 }) |fraction| {
                const policy = cache.createpolicy(primary, secondary, fraction);
                const loaded = cache.loadWithPolicy(.{ .operator = .cache_all_levels }, &out[first], policy);
                expect(out, @src().line, loaded == words[0]);
                const wide = cache.loadWithPolicy(.{ .operator = .cache_global_level }, view(Quad, out, first), policy);
                expect(out, @src().line, sameBytes(wide, @as(Quad, words)));
                const streamed = cache.loadNonCoherentWithPolicy(.cache_streaming, view(u64, out, first), policy);
                expect(out, @src().line, streamed == @as(u64, words[1]) << 32 | words[0]);
                const hinted = cache.loadWithPolicy(.{ .l1_eviction_priority = .evict_last }, &out[first + 2], policy);
                expect(out, @src().line, hinted == words[2]);
            }
        }
    }

    const policy = cache.createpolicy(.evict_first, null, 1.0);
    cache.storeWithPolicy(.{ .operator = .cache_streaming }, &out[first], ~words[0], policy);
    cache.storeWithPolicy(.{ .operator = .write_back }, view(Quad, out, first + 0), @as(Quad, words) + @as(Quad, @splat(1)), policy);
    cache.storeWithPolicy(.{ .l1_eviction_priority = .no_allocate }, &out[first + 3], 5, policy);
    expect(out, @src().line, out[first] == words[0] + 1 and out[first + 1] == words[1] + 1);
    expect(out, @src().line, out[first + 2] == words[2] + 1 and out[first + 3] == 5);
}

export fn cache_range_policy(out: Out) callconv(.kernel) void {
    if (comptime !has_ptx74) return;
    const first = data_word + 4 * tid();
    const words = fillData(out);
    group();
    inline for (primaries) |primary| {
        inline for (secondaries) |secondary| {
            const policy = cache.createpolicyRange(primary, secondary, &out[data_word], 1024, 2048);
            expect(out, @src().line, cache.loadWithPolicy(.{ .operator = .cache_all_levels }, &out[first], policy) == words[0]);
        }
    }
}

export fn cache_hints(out: Out) callconv(.kernel) void {
    if (comptime !has_ptx74) return;
    const first = data_word + 4 * tid();
    const words = fillData(out);
    group();
    cache.prefetch(.l1, &out[first]);
    cache.prefetch(.l2, &out[first]);
    cache.prefetch(.l2_evict_last, &out[first]);
    cache.prefetch(.l2_evict_normal, &out[first]);
    const generic: *u32 = @addrSpaceCast(&out[first]);
    cache.prefetch(.l2, generic);
    cache.prefetchu(generic);
    expect(out, @src().line, out[first] == words[0]);
    group();

    if (tid() == 0) {
        for (0..32) |word| out[1024 - 32 + word] = 7;
        const line: *addrspace(.global) [32]u32 = @ptrCast(@alignCast(out + 1024 - 32));
        cache.applypriority(line);
        expect(out, @src().line, out[1024 - 32] == 7);
        cache.discard(line);
        for (0..32) |word| out[1024 - 32 + word] = 9;
        expect(out, @src().line, out[1024 - 32] == 9 and out[1023] == 9);
    }
}
