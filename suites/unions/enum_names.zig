const h = @import("harness.zig");

const Dense = enum { alpha, beta, gamma, delta_long_name };
const Sparse = enum(u16) { one = 1, ten = 10, hundred = 100, big = 40000 };
const Signed = enum(i8) { neg = -5, zero = 0, pos = 7, min = -128 };
const Open = enum(u8) { a, b, c, _ };
const Tiny = enum(u2) { w, x, y, z };
const Lonely = enum { only };
const Wide = enum(u64) { low = 1, high = 0x1_0000_0000 };
const Union = union(enum) { first: u32, second: u8, third };

const Err = error{ Alpha, Beta, ThreeEee };

fn fnv(s: []const u8) u32 {
    var x: u32 = 0x811c9dc5;
    for (s) |c| x = (x ^ c) *% 16777619;
    return x;
}

inline fn errName(e: anyerror) [:0]const u8 {
    return @errorName(e);
}

fn pick(i: u32) Err {
    return switch (i % 3) {
        0 => error.Alpha,
        1 => error.Beta,
        else => error.ThreeEee,
    };
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    inline for (.{
        .{ Dense, @as(Dense, @enumFromInt(3 -% t)) },
        .{ Dense, @as(Dense, @enumFromInt(t)) },
        .{ Sparse, @as(Sparse, if (t == 0) .hundred else .one) },
        .{ Sparse, @as(Sparse, if (t == 0) .big else .one) },
        .{ Sparse, @as(Sparse, if (t == 0) .ten else .one) },
        .{ Signed, @as(Signed, if (t == 0) .min else .neg) },
        .{ Signed, @as(Signed, if (t == 0) .neg else .pos) },
        .{ Open, @as(Open, @enumFromInt(1 +% t)) },
        .{ Tiny, @as(Tiny, @enumFromInt(3 -% t)) },
        .{ Wide, @as(Wide, if (t == 0) .high else .low) },
    }) |pair| {
        const n = @tagName(pair[1]);
        out[k] = fnv(n);
        k +%= 1;
        out[k] = @intCast(n.len);
        k +%= 1;
    }
    const lonely: Lonely = .only;
    out[k] = @intCast(@tagName(lonely).len);
    k +%= 1;

    const u: Union = if (t == 0) .{ .second = 4 } else .third;
    out[k] = fnv(@tagName(u));
    k +%= 1;
    const un2: Union = if (t == 0) .third else .{ .first = 1 };
    out[k] = fnv(@tagName(un2));
    k +%= 1;

    for (0..3) |i| {
        const e = pick(@intCast(i +% t));
        const n = errName(e);
        out[k] = fnv(n);
        k +%= 1;
        out[k] = @intCast(n.len);
        k +%= 1;
        out[k] = n[n.len];
        k +%= 1;
    }
    const any: anyerror = if (t == 0) error.OutOfMemory else error.Beta;
    out[k] = fnv(errName(any));
    k +%= 1;
    const set: Err = error.Beta;
    out[k] = fnv(@errorName(set));
    k +%= 1;
    out[k] = @intFromBool(@intFromError(error.Beta) == @intFromError(set));
    k +%= 1;
    const back: anyerror = @errorFromInt(@intFromError(any));
    out[k] = fnv(@errorName(back));
    k +%= 1;
    out[k] = @intFromBool(any == error.OutOfMemory);
    k +%= 1;
}

pub const H = h.Harness(run, 38);
comptime {
    _ = H;
}
