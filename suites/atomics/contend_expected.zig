//! The launches of the kernels in contend.zig and the words that each launch must leave in `out`.
//! A null word is not checked.

const std = @import("std");

pub const Launch = struct {
    kernel: []const u8,
    grid: u32,
    block: u32,
    /// Words that the launcher stores in `out` before the launch, as `i<index>=<value>`.
    initial: []const u8 = "",
    words: []const ?u32,
};

const grid = 400;
const block = 253;
const n: u64 = grid * block;

fn low(value: u64) u32 {
    return @truncate(value);
}

fn wide(value: u64) [2]?u32 {
    return .{ low(value), low(value >> 32) };
}

fn neg(value: u64) u32 {
    return low(0 -% value);
}

fn float32(value: f32) u32 {
    return @bitCast(value);
}

fn float64(value: f64) [2]?u32 {
    return wide(@bitCast(value));
}

fn float16(value: f16) u32 {
    return @as(u16, @bitCast(value));
}

fn dupe(arena: std.mem.Allocator, words: []const ?u32) []const ?u32 {
    return arena.dupe(?u32, words) catch @panic("OOM");
}

fn concat(arena: std.mem.Allocator, parts: []const []const ?u32) []const ?u32 {
    return std.mem.concat(arena, ?u32, parts) catch @panic("OOM");
}

fn range(arena: std.mem.Allocator, start: u32, end: u32) []const ?u32 {
    const words = arena.alloc(?u32, end - start) catch @panic("OOM");
    for (words, start..) |*word, value| word.* = @intCast(value);
    return words;
}

fn skipped(arena: std.mem.Allocator, len: usize) []const ?u32 {
    const words = arena.alloc(?u32, len) catch @panic("OOM");
    @memset(words, null);
    return words;
}

pub fn launches(arena: std.mem.Allocator) []const Launch {
    var sum_mod_7: u64 = 0;
    for (0..n) |t| sum_mod_7 += t % 7;
    const b0 = ((n + 2) / 3) % 256;
    const b1 = sum_mod_7 % 256;
    var bytes: u32 = 0;
    for (0..4) |j| bytes |= low(((j + 1) * (n / 4)) % 256) << @intCast(8 * j);
    var halves: u32 = 0;
    for (0..2) |j| halves |= low(((j + 1) * (n / 2)) % 65536) << @intCast(16 * j);
    const blocks = 256;
    const block_len = 256;

    return arena.dupe(Launch, &.{
        .{
            .kernel = "addOrders",
            .grid = grid,
            .block = block,
            .words = dupe(arena, &.{ low(n), low(n), low(n), low(n), low(n), neg(n) }),
        },
        .{
            .kernel = "rmw64",
            .grid = grid,
            .block = block,
            .words = dupe(arena, &(wide(n *% 0x100000001) ++ wide((n - 1) * 3) ++ wide(0 -% (n - 1) * 5) ++
                wide(std.math.maxInt(u64)) ++ wide(n))),
        },
        .{
            .kernel = "rmw32",
            .grid = grid,
            .block = block,
            .initial = "i4=0xFFFFFFFF i5=0xFFFFFFFF i8=0x80000000",
            .words = dupe(arena, &.{ low(n - 1), neg(n - 1), 0xFFFFFFFF, low(n), 0, 0xFFFFFFFF, neg(n), 0xFFFFFFFE, neg(5) }),
        },
        .{
            .kernel = "casCounters",
            .grid = grid,
            .block = block,
            .initial = "i8=0x8000",
            .words = concat(arena, &.{
                &.{ low(n), null },
                &wide(n << 32),
                &.{ low(n % 65536), low(b0 | (b1 << 8)), bytes, halves, 0xFFFD, low(((n + 2) / 3) % 8) },
            }),
        },
        .{ .kernel = "spinLocks", .grid = 128, .block = 64, .words = dupe(arena, &.{ 0, 24576, 0, 24576 }) },
        .{ .kernel = "spinLocksOneLane", .grid = 512, .block = 64, .words = dupe(arena, &.{ 0, 25600 }) },
        .{
            .kernel = "sharedCounters",
            .grid = blocks,
            .block = block_len,
            .words = dupe(arena, &.{
                block_len * blocks,
                255,
                2 * block_len * blocks,
                low(blocks * 0x300),
                blocks * 256,
                block_len / 2 * blocks,
                64 * 4 * blocks,
                128 * 2 * blocks,
                low(0x404040 * blocks),
                low(0x800080 * blocks),
                low(0xFFFFFFFF * blocks),
                block_len * blocks,
                block_len * blocks,
            }),
        },
        .{
            .kernel = "floats",
            .grid = grid,
            .block = block,
            .words = concat(arena, &.{
                &.{ float32(@floatFromInt(n)), null },
                &float64(0.5 * @as(f64, @floatFromInt(n))),
                &.{
                    float32(@floatFromInt(n - 1)),
                    float32(-@as(f32, @floatFromInt(n - 1))),
                    float32(-@as(f32, @floatFromInt(n))),
                    float16(1024.0) | float16(1023.0) << 16,
                    float16(-512.0),
                    null,
                },
                &float64(-@as(f64, @floatFromInt(n - 1))),
                &float64(@floatFromInt(n - 1)),
            }),
        },
        .{ .kernel = "messagePassing", .grid = 64, .block = 32, .words = dupe(arena, &.{ 0, 63 }) },
        .{ .kernel = "exchange", .grid = grid, .block = block, .words = dupe(arena, &.{ 1, low(n - 1) }) },
        .{
            .kernel = "move",
            .grid = 1,
            .block = 1,
            .words = concat(arena, &.{
                skipped(arena, 64),
                &.{ 1, 2 },
                range(arena, 1, 21),
                range(arena, 23, 33),
                skipped(arena, 32),
                range(arena, 3, 23),
                range(arena, 21, 33),
            }),
        },
    }) catch @panic("OOM");
}

/// Formats the initial values and the words as the launcher expects them.
pub fn format(arena: std.mem.Allocator, launch: Launch) []const u8 {
    var text: std.Io.Writer.Allocating = .init(arena);
    text.writer.writeAll(launch.initial) catch @panic("OOM");
    for (launch.words) |word| {
        if (text.written().len != 0) text.writer.writeByte(' ') catch @panic("OOM");
        if (word) |value| text.writer.print("{d}", .{value}) catch @panic("OOM") else text.writer.writeByte('-') catch @panic("OOM");
    }
    return text.written();
}
