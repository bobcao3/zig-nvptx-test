const h = @import("harness.zig");

const Byte = enum(u8) { lo, mid, hi };
const Flags = packed struct(u8) { a: bool, b: bool, c: u6 };

noinline fn addByte(x: u8, y: u8) u8 {
    return x +% y;
}

noinline fn negate(x: i8) i8 {
    return 0 -% x;
}

noinline fn halve(x: u16) u16 {
    return x >> 1;
}

noinline fn invert(x: bool) bool {
    return !x;
}

noinline fn both(x: bool, y: bool) bool {
    return x and y;
}

noinline fn rank(b: Byte) Byte {
    return switch (b) {
        .lo => .hi,
        .mid => .mid,
        .hi => .lo,
    };
}

noinline fn flags(f: Flags) Flags {
    return .{ .a = f.b, .b = f.a, .c = f.c +% 1 };
}

noinline fn callBool(f: *const fn (bool) bool, x: bool) bool {
    return f(x);
}

noinline fn callByte(f: *const fn (u8, u8) u8, x: u8) u8 {
    return f(x, 3);
}

fn run(out: h.Out) void {
    const t: u8 = @truncate(h.tid());
    var k: u32 = 0;
    out[k] = addByte(250 +% t, 10);
    k +%= 1;
    out[k] = @as(u8, @truncate(@as(u32, @bitCast(@as(i32, negate(@as(i8, @bitCast(5 +% t))))))));
    k +%= 1;
    out[k] = halve(0x1235 +% @as(u16, t));
    k +%= 1;
    out[k] = @intFromBool(invert(t == 0));
    k +%= 1;
    out[k] = @intFromBool(both(t == 0, t < 5));
    k +%= 1;
    out[k] = @intFromEnum(rank(@enumFromInt(t)));
    k +%= 1;
    out[k] = @as(u8, @bitCast(flags(@bitCast(@as(u8, 0b1000_0001) +% t))));
    k +%= 1;
    out[k] = @intFromBool(callBool(&invert, t != 0));
    k +%= 1;
    out[k] = callByte(&addByte, 200 +% t);
    k +%= 1;
}

pub const H = h.Harness(run, 9);
comptime {
    _ = H;
}
