const h = @import("harness.zig");

const Pair = extern struct { a: u16, b: u32 };

const Bits = extern union {
    u: u32,
    f: f32,
    bytes: [4]u8,
    halves: [2]u16,
    flag: bool,
};

const Wide = extern union {
    q: u64,
    pair: Pair,
    d: f64,
    small: u8,
};

const Plain = union {
    byte: u8,
    word: u32,
    long: u64,
    pair: Pair,
};

const Packed = packed union {
    u: u32,
    f: f32,
    fields: packed struct { low: u8, high: u24 },
};

const Aligned = union(enum) {
    a: u8,
    b: [4]u32 align(16),
    c: [3]u16,
};

var sink: u32 = 0;

noinline fn useBits(b: Bits) u32 {
    return b.u ^ b.halves[1];
}

noinline fn useWide(w: Wide) u64 {
    return w.q +% w.small;
}

noinline fn makePlain(select: u32) Plain {
    return switch (select) {
        0 => .{ .byte = 0x5a },
        1 => .{ .word = 0xcafef00d },
        else => .{ .long = 0x0123456789abcdef },
    };
}

noinline fn readPlain(p: Plain, select: u32) u64 {
    return switch (select) {
        0 => p.byte,
        1 => p.word,
        else => p.long,
    };
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    var b: Bits = .{ .f = 1.5 + @as(f32, @floatFromInt(t)) };
    out[k] = b.u;
    k +%= 1;
    out[k] = b.bytes[3];
    k +%= 1;
    out[k] = b.halves[0];
    k +%= 1;
    b.bytes[0] = 0xff;
    out[k] = b.u;
    k +%= 1;
    b = .{ .u = 0x80000000 +% t };
    out[k] = @bitCast(b.f);
    k +%= 1;
    out[k] = useBits(b);
    k +%= 1;
    out[k] = @intFromBool(b.flag);
    k +%= 1;

    var w: Wide = .{ .q = 0 };
    w.pair.a = 0x1234;
    w.pair.b = 0x56789abc +% t;
    out[k] = @truncate(w.q);
    k +%= 1;
    out[k] = @truncate(w.q >> 32);
    k +%= 1;
    out[k] = @truncate(useWide(w));
    k +%= 1;
    w.d = 2.0;
    out[k] = @truncate(w.q >> 32);
    k +%= 1;
    w.small = 7;
    out[k] = @truncate(w.q);
    k +%= 1;

    for (0..3) |i| {
        const p = makePlain(@intCast(i +% t));
        out[k] = @truncate(readPlain(p, @intCast(i +% t)));
        k +%= 1;
    }
    var plain: Plain = .{ .long = 0x1111222233334444 };
    out[k] = @truncate(plain.long >> 32);
    k +%= 1;
    plain = .{ .word = 0xaaaabbbb };
    out[k] = plain.word;
    k +%= 1;

    var pk: Packed = .{ .u = 0x3f800000 +% t };
    out[k] = @bitCast(pk.f);
    k +%= 1;
    out[k] = pk.fields.low;
    k +%= 1;
    out[k] = pk.fields.high;
    k +%= 1;
    pk.fields.low = 0x12;
    out[k] = pk.u;
    k +%= 1;
    pk = .{ .fields = .{ .low = 1, .high = @truncate(2 +% t) } };
    out[k] = pk.u;
    k +%= 1;

    var al: Aligned = if (t == 0) .{ .b = .{ 0x0d0e0f10, 0x090a0b0c, 0x05060708, 0x01020304 } } else .{ .a = 1 };
    out[k] = al.b[3];
    k +%= 1;
    out[k] = al.b[0];
    k +%= 1;
    al = .{ .c = .{ 1, 2, 3 +% @as(u16, @truncate(t)) } };
    out[k] = al.c[2];
    k +%= 1;
    out[k] = @backingInt(al);
    k +%= 1;
    sink = k;
}

pub const H = h.Harness(run, 33);
comptime {
    _ = H;
}
