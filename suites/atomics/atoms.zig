const std = @import("std");
const Order = std.lang.AtomicOrder;
const Op = std.lang.AtomicRmwOp;

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

fn Bits(comptime T: type) type {
    return @Int(.unsigned, @bitSizeOf(T));
}

inline fn same(a: anytype, b: @TypeOf(a)) bool {
    return switch (@typeInfo(@TypeOf(a))) {
        .float => (std.math.isNan(a) and std.math.isNan(b)) or @as(Bits(@TypeOf(a)), @bitCast(a)) == @as(Bits(@TypeOf(b)), @bitCast(b)),
        else => a == b,
    };
}

const orders = [_]Order{ .monotonic, .acquire, .release, .acq_rel, .seq_cst };
const raw = [_]u64{ 5, 3, 0x7FFFFFFFFFFFFFFF, 1, 0xA5A5A5A5A5A5A5A5, 0x0F0F0F0F0F0F0F0F, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF };

fn intVal(comptime T: type, comptime v: u64) T {
    const U = @Int(.unsigned, @bitSizeOf(T));
    return @bitCast(@as(U, @truncate(v)));
}

const float_pairs = [_][2]f64{ .{ 1.5, 2.25 }, .{ 3.5, -1.25 }, .{ -0.5, -0.75 }, .{ 1.0, std.math.nan(f64) }, .{ std.math.nan(f64), 2.0 } };

inline fn rmwCase(comptime T: type, comptime op: Op, comptime a: T, comptime b: T, comptime order: Order, p: anytype, st: *State) void {
    const want_old, const want_new = comptime blk: {
        var x: T = a;
        const old = @atomicRmw(T, &x, op, b, .monotonic);
        break :blk .{ old, x };
    };
    p.* = a;
    const old = @atomicRmw(T, p, op, b, order);
    record(st, same(old, want_old) and same(p.*, want_new));
}

const all_ops = [_]Op{ .Xchg, .Add, .Sub, .And, .Nand, .Or, .Xor, .Max, .Min };
const float_ops = [_]Op{ .Xchg, .Add, .Sub, .Max, .Min };

inline fn intCases(comptime T: type, p: anytype, st: *State) void {
    @setEvalBranchQuota(1000000);
    comptime var k: usize = 0;
    inline for (all_ops) |op| {
        inline for (0..raw.len / 2) |i| {
            k += 1;
            rmwCase(T, op, intVal(T, raw[2 * i]), intVal(T, raw[2 * i + 1]), orders[k % orders.len], p, st);
        }
    }
}

inline fn floatCases(comptime T: type, p: anytype, st: *State) void {
    @setEvalBranchQuota(1000000);
    comptime var k: usize = 0;
    inline for (float_ops) |op| {
        inline for (float_pairs) |pair| {
            k += 1;
            rmwCase(T, op, @as(T, @floatCast(pair[0])), @as(T, @floatCast(pair[1])), orders[k % orders.len], p, st);
        }
    }
}

inline fn xchgCase(comptime T: type, comptime a: T, comptime b: T, p: anytype, st: *State) void {
    inline for (orders) |order| rmwCase(T, .Xchg, a, b, order, p, st);
}

const E = enum(u8) { a = 1, b = 7, c = 200 };
const Big = enum(u64) { x = 1 << 40, y = 3 };

inline fn allRmw(p64: anytype, st: *State, comptime as: std.lang.AddressSpace) void {
    @setEvalBranchQuota(1000000);
    inline for (.{ u8, i8, u16, i16, u32, i32, u64, i64, u1, u3, i5, u24, i24, u48, i33 }) |T| {
        const p: *addrspace(as) T = @ptrCast(@alignCast(p64));
        intCases(T, p, st);
    }
    inline for (.{ f16, f32, f64 }) |T| {
        const p: *addrspace(as) T = @ptrCast(@alignCast(p64));
        floatCases(T, p, st);
    }
    {
        const p: *addrspace(as) bool = @ptrCast(@alignCast(p64));
        xchgCase(bool, true, false, p, st);
        xchgCase(bool, false, true, p, st);
    }
    {
        const p: *addrspace(as) E = @ptrCast(@alignCast(p64));
        xchgCase(E, .a, .c, p, st);
    }
    {
        const p: *addrspace(as) Big = @ptrCast(@alignCast(p64));
        xchgCase(Big, .x, .y, p, st);
    }
    {
        const p: *addrspace(as) *u32 = @ptrCast(@alignCast(p64));
        xchgCase(*u32, @ptrFromInt(0x1000), @ptrFromInt(0x2000), p, st);
    }
}

export fn rmwGlobal(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    allRmw(@as(*addrspace(.global) u64, @ptrCast(@alignCast(out + 64))), &st, .global);
    out[0] = st.fails;
    out[1] = st.checks;
    out[2] = st.first;
}

export fn rmwShared(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    allRmw(&shared_buf[0], &st, .shared);
    out[0] = st.fails;
    out[1] = st.checks;
    out[2] = st.first;
}

export fn rmwLocal(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    var slot: u64 = 0;
    inline for (.{ u32, u64, i16, f32 }) |T| {
        var v: T = 0;
        if (@typeInfo(T) == .float) floatCases(T, &v, &st) else intCases(T, &v, &st);
    }
    slot += 1;
    out[0] = st.fails;
    out[1] = st.checks;
    out[2] = st.first;
    out[3] = @intCast(slot);
}

noinline fn viaGeneric(p: *u64, st: *State) void {
    inline for (.{ u8, i16, u32, i64, f32, f64 }) |T| {
        const q: *T = @ptrCast(@alignCast(p));
        if (@typeInfo(T) == .float) floatCases(T, q, st) else intCases(T, q, st);
    }
}

export fn rmwGeneric(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var st: State = .{};
    viaGeneric(@ptrCast(@alignCast(@addrSpaceCast(out + 64))), &st);
    out[0] = st.fails;
    out[1] = st.checks;
    out[2] = st.first;
}

noinline fn bump(p: *u32, q: *u64) u32 {
    const a = @atomicRmw(u32, p, .Add, 5, .seq_cst);
    const b = @cmpxchgStrong(u64, q, 7, 9, .acq_rel, .monotonic);
    const c = @atomicLoad(u32, p, .acquire);
    @atomicStore(u32, p, c + 1, .release);
    return a + (if (b == null) @as(u32, 100) else 0);
}

export fn localGeneric(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    var x: u32 = 10;
    var y: u64 = 7;
    const r = bump(&x, &y);
    out[0] = r;
    out[1] = x;
    out[2] = @intCast(y);
}
