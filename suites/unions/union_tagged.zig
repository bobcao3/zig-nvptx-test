const h = @import("harness.zig");

const Pair = struct { a: u16, b: u32 };

const Shape = union(enum) {
    byte: u8,
    wide: u64,
    float: f32,
    pair: Pair,
    text: []const u8,
    ptr: *const u32,
    none,
};

const Small = union(enum) { a: u8, b: u8, c };

const Tag = @typeInfo(Shape).@"union".tag_type.?;

var global_value: u32 = 0xabcd1234;
const words = [_]u32{ 11, 22, 33 };

fn make(i: u32) Shape {
    return switch (i % 7) {
        0 => .{ .byte = @truncate(i +% 200) },
        1 => .{ .wide = 0x1122334455667788 +% @as(u64, i) },
        2 => .{ .float = @as(f32, @floatFromInt(i)) + 0.5 },
        3 => .{ .pair = .{ .a = @truncate(i *% 1000), .b = 0xdeadbeef +% i } },
        4 => .{ .text = "hello"[0 .. 1 +% i % 5] },
        5 => .{ .ptr = &global_value },
        else => .none,
    };
}

noinline fn describe(s: Shape) u32 {
    return switch (s) {
        .byte => |b| b,
        .wide => |w| @truncate(w >> 8),
        .float => |f| @intFromFloat(f * 2),
        .pair => |p| p.a +% p.b,
        .text => |t| @as(u32, @intCast(t.len)) *% 100 +% t[0],
        .ptr => |p| p.*,
        .none => 0x77,
    };
}

noinline fn pass(s: Shape) Shape {
    return s;
}

noinline fn bump(s: *Shape) void {
    switch (s.*) {
        .byte => |*b| b.* +%= 1,
        .wide => |*w| w.* +%= 1,
        .pair => |*p| p.b +%= 1,
        else => {},
    }
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    for (0..7) |i| {
        const s = make(@intCast(i +% t));
        out[k] = describe(s);
        k +%= 1;
        out[k] = @intFromEnum(@as(Tag, s));
        k +%= 1;
    }
    var s = make(t);
    bump(&s);
    out[k] = describe(s);
    k +%= 1;
    s = pass(make(t +% 1));
    bump(&s);
    out[k] = describe(s);
    k +%= 1;
    s = .{ .pair = .{ .a = 1, .b = 2 } };
    bump(&s);
    out[k] = describe(pass(s));
    k +%= 1;
    s = .{ .wide = 5 };
    s = if (t == 0) .none else s;
    out[k] = describe(s);
    k +%= 1;
    const sm: Small = if (t == 0) .{ .b = 9 } else .c;
    out[k] = switch (sm) {
        .a => 1,
        .b => |v| 100 +% v,
        .c => 3,
    };
    k +%= 1;
    const wp: Shape = .{ .ptr = &words[1 +% t] };
    out[k] = describe(wp);
    k +%= 1;
    out[k] = if (s == .none) 1 else 2;
    k +%= 1;
    const same = make(t +% 3);
    out[k] = @intFromBool(same == Tag.pair);
    k +%= 1;
}

pub const H = h.Harness(run, 20);
comptime {
    _ = H;
}
