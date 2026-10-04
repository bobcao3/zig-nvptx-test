const Pair = extern struct { a: u32, b: u64 };

var bumps: u32 = 0;

export fn ext_add(x: u32, y: u32) u32 {
    return x +% y +% bumps;
}

export fn ext_swap(p: Pair) Pair {
    return .{ .a = @truncate(p.b), .b = p.a };
}

export fn ext_bump() void {
    bumps +%= 1;
}

export fn ext_sum(s: [*]const u32, n: u32) u32 {
    var total: u32 = 0;
    for (0..n) |i| total +%= s[i];
    return total;
}

export fn ext_float(x: f32) f32 {
    return x * 2;
}
