const h = @import("harness.zig");

const Small = error{ Red, Green };

fn run(out: h.Out) void {
    const t = h.tid();
    out[0] = 1;
    const e: anyerror = if (t == 0) error.Blue else error.Red;
    const s: Small = @errorCast(e);
    out[1] = @intFromError(s);
}

pub const H = h.Harness(run, 2);
comptime {
    _ = H;
}
