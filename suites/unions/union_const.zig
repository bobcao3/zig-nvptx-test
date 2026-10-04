const h = @import("harness.zig");

const Pair = struct { a: u16, b: u32 };

const Value = union(enum) {
    int: i16,
    long: u64,
    float: f32,
    pair: Pair,
    text: []const u8,
    ptr: *const u32,
    nothing,
};

const Raw = extern union { u: u32, b: [4]u8, p: ?*const u32 };

const Holder = struct {
    tag: u8,
    value: Value,
};

var counter: u32 = 0x1000;
const answer: u32 = 42;

const table = [_]Value{
    .{ .int = -5 },
    .{ .long = 0xfedcba9876543210 },
    .{ .float = 2.5 },
    .{ .pair = .{ .a = 7, .b = 0x01020304 } },
    .{ .text = "tag" },
    .{ .ptr = &answer },
    .{ .ptr = &counter },
    .nothing,
};

const raws = [_]Raw{ .{ .u = 0x11223344 }, .{ .b = .{ 9, 8, 7, 6 } }, .{ .p = &answer }, .{ .p = null } };

var holders = [_]Holder{
    .{ .tag = 1, .value = .{ .text = "xyz" } },
    .{ .tag = 2, .value = .{ .ptr = &counter } },
};

var global_value: Value = .{ .pair = .{ .a = 3, .b = 4 } };

fn digest(v: Value) u32 {
    return switch (v) {
        .int => |i| @bitCast(@as(i32, i)),
        .long => |l| @truncate(l >> 4),
        .float => |f| @bitCast(f),
        .pair => |p| @as(u32, p.a) << 16 | (p.b & 0xffff),
        .text => |s| @as(u32, s[0]) << 8 | s[2],
        .ptr => |p| p.*,
        .nothing => 0x99,
    };
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    for (0..table.len) |i| {
        out[k] = digest(table[i +% t]);
        k +%= 1;
    }
    out[k] = digest(.{ .ptr = &answer });
    k +%= 1;
    out[k] = digest(global_value);
    k +%= 1;
    global_value = table[3 +% t];
    out[k] = digest(global_value);
    k +%= 1;
    out[k] = digest(holders[t].value) +% holders[t].tag;
    k +%= 1;
    out[k] = digest(holders[1 +% t].value) +% holders[1 +% t].tag;
    k +%= 1;
    const raw = raws[t];
    out[k] = raw.u;
    k +%= 1;
    out[k] = raws[1 +% t].u;
    k +%= 1;
    out[k] = raws[2 +% t].p.?.*;
    k +%= 1;
    out[k] = @intFromBool(raws[3 +% t].p == null);
    k +%= 1;
    const local: Value = if (t == 0) .{ .text = "q" ++ "r" ++ "s" } else .nothing;
    out[k] = digest(local);
    k +%= 1;
}

pub const H = h.Harness(run, 18);
comptime {
    _ = H;
}
