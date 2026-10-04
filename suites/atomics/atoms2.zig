const std = @import("std");
const Order = std.lang.AtomicOrder;

var shared_buf: [16]u64 align(8) addrspace(.shared) = undefined;

const State = struct {
    checks: u32 = 0,
    fails: u32 = 0,
    first: u32 = 0,
};

inline fn record(st: *State, ok: bool) void {
    st.checks += 1;
    if (!ok) {
        st.fails += 1;
        if (st.first == 0) st.first = st.checks;
    }
}

const raw = [_]u64{ 5, 3, 0x7FFFFFFFFFFFFFFF, 1, 0xA5A5A5A5A5A5A5A5, 0x0F0F0F0F0F0F0F0F, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF };

fn intVal(comptime T: type, comptime v: u64) T {
    const U = @Int(.unsigned, @bitSizeOf(T));
    return @bitCast(@as(U, @truncate(v)));
}

const Pair = struct { Order, Order };
const cas_orders = [_]Pair{
    .{ .monotonic, .monotonic }, .{ .acquire, .monotonic }, .{ .release, .monotonic },
    .{ .acq_rel, .acquire },     .{ .seq_cst, .seq_cst },   .{ .release, .acquire },
    .{ .acq_rel, .monotonic },   .{ .seq_cst, .acquire },   .{ .acquire, .acquire },
    .{ .seq_cst, .monotonic },
};

inline fn casCase(comptime T: type, comptime a: T, comptime b: T, comptime pair: Pair, comptime weak: bool, p: anytype, st: *State) void {
    // success: memory equals expected
    p.* = a;
    const r1 = if (weak) @cmpxchgWeak(T, p, a, b, pair[0], pair[1]) else @cmpxchgStrong(T, p, a, b, pair[0], pair[1]);
    record(st, r1 == null and p.* == b);
    // failure: memory differs from expected, so the old value comes back
    if (a != b) {
        p.* = a;
        const r2 = @cmpxchgStrong(T, p, b, a, pair[0], pair[1]);
        record(st, r2 != null and r2.? == a and p.* == a);
    }
}

inline fn casAll(comptime T: type, p: anytype, st: *State) void {
    comptime var k: usize = 0;
    inline for (0..raw.len / 2) |i| {
        k += 1;
        const pair = cas_orders[k % cas_orders.len];
        casCase(T, intVal(T, raw[2 * i]), intVal(T, raw[2 * i + 1]), pair, false, p, st);
        casCase(T, intVal(T, raw[2 * i]), intVal(T, raw[2 * i + 1]), pair, true, p, st);
    }
}

const E = enum(u8) { a = 1, b = 7, c = 200 };

inline fn all(p64: anytype, st: *State, comptime as: std.lang.AddressSpace) void {
    @setEvalBranchQuota(1000000);
    inline for (.{ u8, i8, u16, i16, u32, i32, u64, i64, u1, u3, i5, u24, i24, u48, i33 }) |T| {
        const p: *addrspace(as) T = @ptrCast(@alignCast(p64));
        casAll(T, p, st);
    }
    {
        const p: *addrspace(as) bool = @ptrCast(@alignCast(p64));
        inline for (cas_orders) |pair| casCase(bool, true, false, pair, false, p, st);
    }
    {
        const p: *addrspace(as) E = @ptrCast(@alignCast(p64));
        inline for (cas_orders) |pair| casCase(E, .a, .c, pair, false, p, st);
    }
    {
        const p: *addrspace(as) *u32 = @ptrCast(@alignCast(p64));
        inline for (cas_orders) |pair| casCase(*u32, @ptrFromInt(0x1000), @ptrFromInt(0x2000), pair, false, p, st);
        const q: *addrspace(as) ?*u32 = @ptrCast(@alignCast(p64));
        q.* = @ptrFromInt(0x40);
        const r = @cmpxchgStrong(?*u32, q, @ptrFromInt(0x40), null, .seq_cst, .seq_cst);
        record(st, r == null and q.* == null);
        const r2 = @cmpxchgStrong(?*u32, q, @ptrFromInt(0x40), null, .seq_cst, .seq_cst);
        record(st, r2 != null and r2.? == null and q.* == null);
    }
}

inline fn loadStore(comptime T: type, p: anytype, st: *State) void {
    inline for (.{ Order.unordered, .monotonic, .release, .seq_cst }) |so| {
        inline for (.{ Order.unordered, .monotonic, .acquire, .seq_cst }) |lo| {
            inline for (0..raw.len) |i| {
                const v = comptime intVal(T, raw[i]);
                switch (so) {
                    .unordered => @atomicStore(T, p, v, .unordered),
                    .monotonic => @atomicStore(T, p, v, .monotonic),
                    .release => @atomicStore(T, p, v, .release),
                    else => @atomicStore(T, p, v, .seq_cst),
                }
                const got = switch (lo) {
                    .unordered => @atomicLoad(T, p, .unordered),
                    .monotonic => @atomicLoad(T, p, .monotonic),
                    .acquire => @atomicLoad(T, p, .acquire),
                    else => @atomicLoad(T, p, .seq_cst),
                };
                record(st, got == v and p.* == v);
            }
        }
    }
}

inline fn allLoadStore(p64: anytype, st: *State, comptime as: std.lang.AddressSpace) void {
    @setEvalBranchQuota(1000000);
    inline for (.{ u8, i8, u16, i16, u32, i32, u64, i64, u1, u3, i5, u24, i24, u48, i33 }) |T| {
        const p: *addrspace(as) T = @ptrCast(@alignCast(p64));
        loadStore(T, p, st);
    }
    inline for (.{ f16, f32, f64 }) |T| {
        const p: *addrspace(as) T = @ptrCast(@alignCast(p64));
        @atomicStore(T, p, 2.5, .seq_cst);
        record(st, @atomicLoad(T, p, .seq_cst) == 2.5);
        @atomicStore(T, p, -0.125, .release);
        record(st, @atomicLoad(T, p, .acquire) == -0.125);
    }
    {
        const p: *addrspace(as) bool = @ptrCast(@alignCast(p64));
        @atomicStore(bool, p, true, .seq_cst);
        record(st, @atomicLoad(bool, p, .seq_cst));
        @atomicStore(bool, p, false, .release);
        record(st, !@atomicLoad(bool, p, .acquire));
        @atomicStore(bool, p, true, .unordered);
        record(st, @atomicLoad(bool, p, .unordered));
    }
    {
        const p: *addrspace(as) E = @ptrCast(@alignCast(p64));
        @atomicStore(E, p, .c, .seq_cst);
        record(st, @atomicLoad(E, p, .acquire) == .c);
    }
    {
        const p: *addrspace(as) *u32 = @ptrCast(@alignCast(p64));
        @atomicStore(*u32, p, @ptrFromInt(0x3000), .release);
        record(st, @intFromPtr(@atomicLoad(*u32, p, .seq_cst)) == 0x3000);
    }
}

inline fn report(out: [*]addrspace(.global) u32, st: State) void {
    out[0] = st.fails;
    out[1] = st.checks;
    out[2] = st.first;
}

export fn casGlobal(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    all(@as(*addrspace(.global) u64, @ptrCast(@alignCast(out + 64))), &st, .global);
    report(out, st);
}

export fn casShared(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    all(&shared_buf[0], &st, .shared);
    report(out, st);
}

export fn lsGlobal(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    allLoadStore(@as(*addrspace(.global) u64, @ptrCast(@alignCast(out + 64))), &st, .global);
    report(out, st);
}

export fn lsShared(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    allLoadStore(&shared_buf[0], &st, .shared);
    report(out, st);
}

noinline fn viaGeneric(p: *u64, st: *State) void {
    all(p, st, .generic);
    allLoadStore(p, st, .generic);
}

export fn casGeneric(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    viaGeneric(@ptrCast(@alignCast(@addrSpaceCast(out + 64))), &st);
    report(out, st);
}

export fn casLocal(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    var v: u64 = 0;
    all(&v, &st, .generic);
    allLoadStore(&v, &st, .generic);
    report(out, st);
}
