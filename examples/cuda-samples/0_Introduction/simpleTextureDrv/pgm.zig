// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! Loading and saving of binary PGM images.

const std = @import("std");

pub const Image = struct {
    width: u32,
    height: u32,
    pixels: []f32,
};

/// Parses a binary PGM file ("P5") into pixels in [0, 1].
pub fn load(gpa: std.mem.Allocator, bytes: []const u8) !Image {
    if (!std.mem.startsWith(u8, bytes, "P5")) return error.NotPgm;
    var cursor: usize = 2;
    var header: [3]u32 = undefined;
    for (&header) |*number| {
        while (cursor < bytes.len and (std.ascii.isWhitespace(bytes[cursor]) or bytes[cursor] == '#')) {
            if (bytes[cursor] == '#') {
                cursor = std.mem.indexOfScalarPos(u8, bytes, cursor, '\n') orelse return error.TruncatedPgm;
            }
            cursor += 1;
        }
        const start = cursor;
        while (cursor < bytes.len and std.ascii.isDigit(bytes[cursor])) cursor += 1;
        number.* = try std.fmt.parseInt(u32, bytes[start..cursor], 10);
    }
    const width, const height, _ = header;
    const texels = bytes[cursor + 1 ..];
    if (texels.len < width * height) return error.TruncatedPgm;

    const pixels = try gpa.alloc(f32, width * height);
    for (pixels, texels) |*pixel, texel| pixel.* = @as(f32, @floatFromInt(texel)) / 255.0;
    return .{ .width = width, .height = height, .pixels = pixels };
}

/// Returns a binary PGM file with the pixels scaled from [0, 1] to bytes.
pub fn save(gpa: std.mem.Allocator, image: Image) ![]u8 {
    var file: std.ArrayList(u8) = .empty;
    errdefer file.deinit(gpa);
    try file.print(gpa, "P5\n{d}\n{d}\n255\n", .{ image.width, image.height });
    for (image.pixels) |pixel| {
        try file.append(gpa, @intFromFloat(std.math.clamp(pixel * 255.0, 0, 255)));
    }
    return file.toOwnedSlice(gpa);
}
