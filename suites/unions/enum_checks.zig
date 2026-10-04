const h = @import("harness.zig");

const Dense = enum(u8) { a, b, c, d };
const Sparse = enum(i16) { m = -300, z = 0, p = 5, q = 12345 };
const Tiny = enum(u3) { a, b, c = 5 };
const One = enum { only };
const Small = error{ Red, Green };
const Big = error{ Red, Green, Blue, Yellow };

fn fnv(x: u32, s: []const u8) u32 {
    var v = x;
    for (s) |c| v = (v ^ c) *% 16777619;
    return v;
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    const d: Dense = @enumFromInt(2 +% t);
    out[k] = @intFromEnum(d);
    k +%= 1;
    const s: Sparse = @enumFromInt(@as(i16, if (t == 0) -300 else 5));
    out[k] = @bitCast(@as(i32, @intFromEnum(s)));
    k +%= 1;
    const s2: Sparse = @enumFromInt(@as(i16, if (t == 0) 12345 else 0));
    out[k] = @bitCast(@as(i32, @intFromEnum(s2)));
    k +%= 1;
    const tn: Tiny = @enumFromInt(@as(u3, @truncate(5 -% t *% 5)));
    out[k] = @intFromEnum(tn);
    k +%= 1;
    const one: One = @enumFromInt(0);
    out[k] = fnv(0, @tagName(one));
    k +%= 1;
    const wide: u32 = if (t == 0) 3 else 1;
    const d2: Dense = @enumFromInt(wide);
    out[k] = fnv(0, @tagName(d2));
    k +%= 1;

    const big: Big = if (t == 0) error.Blue else error.Green;
    const small: Small = @errorCast(@as(Big, if (t == 0) error.Red else error.Green));
    out[k] = fnv(0, @errorName(small));
    k +%= 1;
    const any: anyerror = big;
    const back: Big = @errorCast(any);
    out[k] = fnv(0, @errorName(back));
    k +%= 1;
    const code = @intFromError(any);
    const again: anyerror = @errorFromInt(code);
    out[k] = fnv(0, @errorName(again));
    k +%= 1;
    const wide_err: anyerror = if (t == 0) error.Yellow else error.Red;
    const narrowed: Big = @errorCast(wide_err);
    out[k] = fnv(0, @errorName(narrowed));
    k +%= 1;
}

pub const H = h.Harness(run, 10);
comptime {
    _ = H;
}
