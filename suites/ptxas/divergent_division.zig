//! The kernel divides by a constant on one side of a divergent branch and calls a function on the
//! other side. A failed check stores its line number in `out[0]`.

const Out = [*]addrspace(.global) u32;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

noinline fn hide(x: u64) u64 {
    return x;
}

noinline fn alwaysFalse() bool {
    return false;
}

fn mix(h: u32, x: u64) u32 {
    const a = (h ^ @as(u32, @truncate(x))) *% 0x01000193;
    return (a ^ @as(u32, @truncate(x >> 32))) *% 0x01000193;
}

fn walk(v: u64) u32 {
    var h: u32 = 0;
    const low: u30 = @truncate(v);
    const lane: u2 = @truncate(low);
    for (0..3) |i| {
        switch (lane) {
            0 => {
                h = mix(h, @intFromBool(alwaysFalse()));
                h = mix(h, @as(u32, @intCast(i)));
            },
            else => h = mix(h, low / 1073741822),
        }
        h = mix(h, @intFromBool(alwaysFalse()));
    }
    return h;
}

export fn divergent_division(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const even = comptime walk(0);
    const odd = comptime walk(0xffff_ffff);
    const h = walk(hide(if (t % 2 == 0) 0 else 0xffff_ffff));
    expect(out, @src().line, h == if (t % 2 == 0) even else odd);
}
