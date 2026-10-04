//! Encodes 8-bit RGBA images as PNG (https://www.w3.org/TR/png-3/) with filter type 0 (none).

const std = @import("std");
const flate = std.compress.flate;
const Rgba8 = @import("scene").Rgba8;

pub fn encode(gpa: std.mem.Allocator, width: u32, height: u32, pixels: []const Rgba8) ![]u8 {
    std.debug.assert(pixels.len == @as(usize, width) * height);
    var file: std.Io.Writer.Allocating = try .initCapacity(gpa, 1 << 16);
    errdefer file.deinit();
    try file.writer.writeAll("\x89PNG\r\n\x1a\n");

    var header: [13]u8 = undefined;
    std.mem.writeInt(u32, header[0..4], width, .big);
    std.mem.writeInt(u32, header[4..8], height, .big);
    // Bit depth 8, color type 6 (RGBA), compression method 0, filter method 0, no interlace.
    header[8..].* = .{ 8, 6, 0, 0, 0 };
    try writeChunk(&file.writer, "IHDR", &header);

    var data: std.Io.Writer.Allocating = try .initCapacity(gpa, 1 << 16);
    defer data.deinit();
    var window: [flate.max_window_len]u8 = undefined;
    var compress: flate.Compress = try .init(&data.writer, &window, .zlib, .default);
    for (0..height) |y| {
        try compress.writer.writeByte(0);
        for (pixels[y * width ..][0..width]) |pixel| {
            try compress.writer.writeAll(&.{ pixel.r, pixel.g, pixel.b, pixel.a });
        }
    }
    try compress.finish();
    try writeChunk(&file.writer, "IDAT", data.written());
    try writeChunk(&file.writer, "IEND", "");
    return file.toOwnedSlice();
}

fn writeChunk(writer: *std.Io.Writer, kind: *const [4]u8, data: []const u8) !void {
    try writer.writeInt(u32, @intCast(data.len), .big);
    try writer.writeAll(kind);
    try writer.writeAll(data);
    var crc: std.hash.Crc32 = .init();
    crc.update(kind);
    crc.update(data);
    try writer.writeInt(u32, crc.final(), .big);
}
