const h = @import("harness.zig");

const Sparse = enum(u8) { a = 1, b = 7, c = 200 };

fn run(out: h.Out) void {
    const t = h.tid();
    out[0] = 1;
    const s: Sparse = @enumFromInt(@as(u8, @truncate(8 +% t)));
    out[1] = @intFromEnum(s);
}

pub const H = h.Harness(run, 2);
comptime {
    _ = H;
}
