const h = @import("harness.zig");

const Pair = extern struct { a: u32, b: u64 };

extern fn ext_add(x: u32, y: u32) u32;
extern fn ext_swap(p: Pair) Pair;
extern fn ext_bump() void;
extern fn ext_sum(s: [*]const u32, n: u32) u32;
extern fn ext_float(x: f32) f32;

var holder: *const fn (u32, u32) callconv(h.device_cc) u32 = &ext_add;
const words = [_]u32{ 10, 20, 30 };

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    out[k] = ext_add(40 +% t, 2);
    k +%= 1;
    const swapped = ext_swap(.{ .a = 7 +% t, .b = 0x200000001 });
    out[k] = swapped.a;
    k +%= 1;
    out[k] = @truncate(swapped.b);
    k +%= 1;
    ext_bump();
    ext_bump();
    out[k] = ext_add(0, 0);
    k +%= 1;
    out[k] = ext_sum(&words, 3 -% t);
    k +%= 1;
    out[k] = @bitCast(ext_float(1.5 + @as(f32, @floatFromInt(t))));
    k +%= 1;
    out[k] = holder(100, 5 +% t);
    k +%= 1;
    const local: *const fn (u32, u32) callconv(h.device_cc) u32 = &ext_add;
    out[k] = local(1, 2 +% t);
    k +%= 1;
}

pub const H = h.Harness(run, 9);
comptime {
    _ = H;
}
