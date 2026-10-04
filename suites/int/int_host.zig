const std = @import("std");
const ops = @import("int_ops.zig");

pub fn main(init: std.process.Init) !void {
    var args = init.minimal.args.iterate();
    _ = args.next();
    const name = args.next().?;
    var words: [1024]u32 = @splat(0xdeadbeef);
    inline for (ops.names) |n| {
        if (std.mem.eql(u8, name, n)) @field(ops, n)(&words);
    }
    var buf: [16384]u8 = undefined;
    var w = std.Io.File.stdout().writer(init.io, &buf);
    for (words) |x| {
        if (x == 0xdeadbeef) break;
        try w.interface.print("{d} ", .{x});
    }
    try w.interface.flush();
}
