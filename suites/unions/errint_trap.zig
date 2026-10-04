const h = @import("harness.zig");

fn run(out: h.Out) void {
    const t = h.tid();
    out[0] = 1;
    const valid: anyerror = @errorFromInt(@as(u16, @truncate(1 +% t)));
    out[1] = @intFromError(valid);
    const bad: anyerror = @errorFromInt(@as(u16, @truncate(30000 +% t)));
    out[2] = @intFromError(bad);
}

pub const H = h.Harness(run, 3);
comptime {
    _ = H;
}
