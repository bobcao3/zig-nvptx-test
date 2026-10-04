const h = @import("harness.zig");

noinline fn returnAddress() usize {
    return @returnAddress();
}

noinline fn frameAddress() usize {
    return @frameAddress();
}

fn run(out: h.Out) void {
    const t = h.tid();
    out[0] = if (h.gpu) @intFromBool(returnAddress() +% t == 0) else 1;
    out[1] = if (h.gpu) @intFromBool(frameAddress() +% t == 0) else 1;
    out[2] = if (h.gpu) @intFromBool(@returnAddress() == 0) else 1;
}

pub const H = h.Harness(run, 3);
comptime {
    _ = H;
}
