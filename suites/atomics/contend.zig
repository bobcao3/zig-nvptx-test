const std = @import("std");

var sh_words: [8]u32 addrspace(.shared) = undefined;
var sh_f32: f32 addrspace(.shared) = undefined;
var sh_f64: f64 addrspace(.shared) = undefined;
var sh_f16: f16 addrspace(.shared) = undefined;
var sh_u64: u64 addrspace(.shared) = undefined;
var sh_bytes: [8]u8 addrspace(.shared) = undefined;
var sh_halves: [4]u16 addrspace(.shared) = undefined;

fn gtid() u32 {
    return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
}

fn at(comptime T: type, out: [*]addrspace(.global) u32, word: usize) *addrspace(.global) T {
    return @ptrCast(@alignCast(out + word));
}

fn barrier() void {
    std.nvptx.barrierSyncAligned(.work_group, null);
}

/// Every thread adds 1 with each order.
export fn addOrders(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    _ = @atomicRmw(u32, &out[0], .Add, 1, .monotonic);
    _ = @atomicRmw(u32, &out[1], .Add, 1, .acquire);
    _ = @atomicRmw(u32, &out[2], .Add, 1, .release);
    _ = @atomicRmw(u32, &out[3], .Add, 1, .acq_rel);
    _ = @atomicRmw(u32, &out[4], .Add, 1, .seq_cst);
    _ = @atomicRmw(i32, at(i32, out, 5), .Sub, 1, .seq_cst);
}

export fn rmw64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const t: u64 = gtid();
    _ = @atomicRmw(u64, at(u64, out, 0), .Add, 0x100000001, .monotonic);
    _ = @atomicRmw(i64, at(i64, out, 2), .Max, @intCast(t * 3), .acq_rel);
    _ = @atomicRmw(i64, at(i64, out, 4), .Min, -@as(i64, @intCast(t * 5)), .seq_cst);
    _ = @atomicRmw(u64, at(u64, out, 6), .Or, @as(u64, 1) << @intCast(t % 64), .monotonic);
    _ = @atomicRmw(u64, at(u64, out, 8), .Xor, t + 1, .monotonic);
}

export fn rmw32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const t = gtid();
    _ = @atomicRmw(u32, &out[0], .Max, t, .monotonic);
    _ = @atomicRmw(i32, at(i32, out, 1), .Min, -@as(i32, @intCast(t)), .monotonic);
    _ = @atomicRmw(u32, &out[2], .Or, @as(u32, 1) << @intCast(t % 32), .monotonic);
    _ = @atomicRmw(u32, &out[3], .Xor, t + 1, .monotonic);
    _ = @atomicRmw(u32, &out[4], .And, ~(@as(u32, 1) << @intCast(t % 32)), .monotonic);
    _ = @atomicRmw(u32, &out[5], .Nand, 0xFFFF0000, .monotonic);
    _ = @atomicRmw(u32, &out[6], .Sub, 1, .monotonic);
    _ = @atomicRmw(u32, &out[7], .Max, 0xFFFFFFFE - t, .monotonic);
    _ = @atomicRmw(i32, at(i32, out, 8), .Max, -@as(i32, @intCast(t)) - 5, .monotonic);
}

/// Compare and swap counters of every width.
export fn casCounters(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const t = gtid();
    {
        const p: *addrspace(.global) u32 = &out[0];
        var old = @atomicLoad(u32, p, .monotonic);
        while (@cmpxchgWeak(u32, p, old, old + 1, .acq_rel, .monotonic)) |cur| old = cur;
    }
    {
        const p: *addrspace(.global) u64 = @ptrCast(@alignCast(&out[2]));
        var old = @atomicLoad(u64, p, .monotonic);
        while (@cmpxchgStrong(u64, p, old, old + 0x100000000, .seq_cst, .seq_cst)) |cur| old = cur;
    }
    {
        const p: *addrspace(.global) u16 = @ptrCast(&out[4]);
        var old = @atomicLoad(u16, p, .monotonic);
        while (@cmpxchgStrong(u16, p, old, old +% 1, .acq_rel, .acquire)) |cur| old = cur;
    }
    if (t % 3 == 0) {
        // The byte shares a 16-bit word with a neighbour that other threads update.
        const p: *addrspace(.global) u8 = @ptrCast(&out[5]);
        var old = @atomicLoad(u8, p, .monotonic);
        while (@cmpxchgWeak(u8, p, old, old +% 1, .release, .monotonic)) |cur| old = cur;
    }
    {
        const p: *addrspace(.global) u8 = @ptrCast(@as([*]addrspace(.global) u8, @ptrCast(&out[5])) + 1);
        _ = @atomicRmw(u8, p, .Add, @intCast(t % 7), .monotonic);
    }
    {
        const q: [*]addrspace(.global) u8 = @ptrCast(&out[6]);
        _ = @atomicRmw(u8, &q[t % 4], .Add, @intCast(t % 4 + 1), .monotonic);
    }
    {
        const q: [*]addrspace(.global) u16 = @ptrCast(&out[7]);
        _ = @atomicRmw(u16, &q[t % 2], .Add, @intCast(t % 2 + 1), .acq_rel);
    }
    {
        const p: *addrspace(.global) i16 = @ptrCast(&out[8]);
        _ = @atomicRmw(i16, p, .Max, -@as(i16, @intCast(t % 1000)) - 3, .monotonic);
    }
    if (t % 3 == 0) {
        const p: *addrspace(.global) u3 = @ptrCast(&out[9]);
        _ = @atomicRmw(u3, p, .Add, 1, .monotonic);
    }
}

fn lockCasThenCount(lock: anytype, counter: anytype) void {
    while (@cmpxchgWeak(u32, lock, 0, 1, .acquire, .monotonic) != null) {}
    counter.* += 1;
    @atomicStore(u32, lock, 0, .release);
}

fn lockXchgThenCount(lock: anytype, counter: anytype) void {
    while (@atomicRmw(u32, lock, .Xchg, 1, .acquire) != 0) {}
    counter.* += 1;
    @atomicStore(u32, lock, 0, .release);
}

/// A spin lock built from cmpxchg or xchg protects a plain counter.
export fn spinLocks(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    inline for (0..3) |_| {
        lockCasThenCount(&out[0], &out[1]);
        lockXchgThenCount(&out[2], &out[3]);
    }
}

/// One thread of each block takes part so that blocks contend, the warp lanes do not spin on each other.
export fn spinLocksOneLane(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    if (@workItemId(0) == 0) {
        for (0..50) |_| lockCasThenCount(&out[0], &out[1]);
    }
}

/// Every thread adds to words in shared memory. Thread 0 of each block adds the block totals to global memory.
export fn sharedCounters(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const t = @workItemId(0);
    if (t == 0) {
        sh_words = .{ 0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0 };
        sh_u64 = 0;
        sh_f32 = 0;
        sh_f64 = 0;
        sh_f16 = 0;
        sh_bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
        sh_halves = .{ 0, 0, 0, 0 };
    }
    barrier();
    _ = @atomicRmw(u32, &sh_words[0], .Add, 1, .monotonic);
    _ = @atomicRmw(u32, &sh_words[1], .Max, t, .acq_rel);
    {
        var old = @atomicLoad(u32, &sh_words[2], .acquire);
        while (@cmpxchgWeak(u32, &sh_words[2], old, old + 2, .seq_cst, .monotonic)) |cur| old = cur;
    }
    _ = @atomicRmw(u64, &sh_u64, .Add, 0x100000003, .seq_cst);
    _ = @atomicRmw(f32, &sh_f32, .Add, 0.5, .monotonic);
    _ = @atomicRmw(f64, &sh_f64, .Add, 0.25, .monotonic);
    _ = @atomicRmw(f16, &sh_f16, .Add, 0.5, .monotonic);
    _ = @atomicRmw(u8, &sh_bytes[t % 4], .Add, 1, .monotonic);
    _ = @atomicRmw(u16, &sh_halves[t % 2], .Add, 1, .release);
    _ = @atomicRmw(u32, &sh_words[3], .Nand, 0xFFFFFFFF, .monotonic);
    _ = @atomicRmw(u32, &sh_words[4], .Xor, t + 1, .monotonic);
    {
        // lock in shared memory, protecting a plain counter
        lockCasThenCount(&sh_words[5], &sh_words[6]);
    }
    barrier();
    if (t == 0) {
        _ = @atomicRmw(u32, &out[0], .Add, sh_words[0], .monotonic);
        _ = @atomicRmw(u32, &out[1], .Max, sh_words[1], .monotonic);
        _ = @atomicRmw(u32, &out[2], .Add, sh_words[2], .monotonic);
        _ = @atomicRmw(u32, &out[3], .Add, @truncate(sh_u64), .monotonic);
        _ = @atomicRmw(u32, &out[4], .Add, @truncate(sh_u64 >> 32), .monotonic);
        _ = @atomicRmw(u32, &out[5], .Add, @intFromFloat(sh_f32), .monotonic);
        _ = @atomicRmw(u32, &out[6], .Add, @intFromFloat(sh_f64 * 4), .monotonic);
        _ = @atomicRmw(u32, &out[7], .Add, @intFromFloat(sh_f16 * 2), .monotonic);
        _ = @atomicRmw(u32, &out[8], .Add, @as(u32, sh_bytes[0]) + @as(u32, sh_bytes[1]) * 256 + @as(u32, sh_bytes[2]) * 65536, .monotonic);
        _ = @atomicRmw(u32, &out[9], .Add, @as(u32, sh_halves[0]) + sh_halves[1] * @as(u32, 65536), .monotonic);
        _ = @atomicRmw(u32, &out[10], .Add, sh_words[3], .monotonic);
        _ = @atomicRmw(u32, &out[11], .Add, sh_words[4], .monotonic);
        _ = @atomicRmw(u32, &out[12], .Add, sh_words[6], .monotonic);
    }
}

export fn floats(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const t = gtid();
    const f: f32 = @floatFromInt(t);
    _ = @atomicRmw(f32, at(f32, out, 0), .Add, 1.0, .monotonic);
    _ = @atomicRmw(f64, at(f64, out, 2), .Add, 0.5, .acq_rel);
    _ = @atomicRmw(f32, at(f32, out, 4), .Max, f, .monotonic);
    _ = @atomicRmw(f32, at(f32, out, 5), .Min, -f, .monotonic);
    _ = @atomicRmw(f32, at(f32, out, 6), .Sub, 1.0, .seq_cst);
    if (t < 1024) {
        _ = @atomicRmw(f16, at(f16, out, 7), .Add, 1.0, .monotonic);
        _ = @atomicRmw(f16, &@as([*]addrspace(.global) f16, @ptrCast(out + 7))[1], .Max, @floatFromInt(t), .monotonic);
        _ = @atomicRmw(f16, at(f16, out, 8), .Sub, 0.5, .monotonic);
    }
    _ = @atomicRmw(f64, at(f64, out, 10), .Min, -@as(f64, @floatFromInt(t)), .monotonic);
    _ = @atomicRmw(f64, at(f64, out, 12), .Max, @as(f64, @floatFromInt(t)), .monotonic);
}

/// Message passing: block 0 publishes data with a release store, the other blocks read it after an acquire load.
export fn messagePassing(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const data: *addrspace(.global) u32 = &out[16];
    const flag: *addrspace(.global) u32 = &out[17];
    if (@workGroupId(0) == 0) {
        if (@workItemId(0) == 0) {
            data.* = 42;
            @atomicStore(u32, flag, 1, .release);
        }
    } else if (@workItemId(0) == 0) {
        while (@atomicLoad(u32, flag, .acquire) == 0) {}
        if (data.* != 42) _ = @atomicRmw(u32, &out[0], .Add, 1, .monotonic);
        _ = @atomicRmw(u32, &out[1], .Add, 1, .monotonic);
    }
}

/// Every thread swaps its id into one word. The swapped-out values and the final value are a permutation of the
/// ids and the initial zero, so their sum is fixed. The last thread to finish checks the sum.
export fn exchange(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    const t = gtid();
    const n = @workGroupSize(0) * 400;
    const old = @atomicRmw(u32, &out[4], .Xchg, t + 1, .acq_rel);
    _ = @atomicRmw(u64, at(u64, out, 6), .Add, old, .monotonic);
    if (old != 0) _ = @atomicRmw(u32, &out[1], .Add, 1, .monotonic);
    const done = @atomicRmw(u32, &out[8], .Add, 1, .acq_rel);
    if (done == n - 1) {
        const total = @atomicLoad(u64, at(u64, out, 6), .acquire) + @atomicLoad(u32, &out[4], .acquire);
        const want: u64 = @as(u64, n) * (n + 1) / 2;
        out[0] = @intFromBool(total == want);
    }
}

/// A memmove of overlapping ranges in global memory.
export fn move(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    if (gtid() != 0) return;
    for (0..32) |i| out[64 + i] = @intCast(i + 1);
    @memmove(out[66..][0..20], out[64..][0..20]);
    for (0..32) |i| out[128 + i] = @intCast(i + 1);
    @memmove(out[128..][0..20], out[130..][0..20]);
}
