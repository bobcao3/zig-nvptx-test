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

//! Contains basic image operations implementation.
//!
//! This file contains implementation of basic bitmap loading, saving, conversions to different
//! representations and memory management routines.

const std = @import("std");

const bmp_signature = 0x4D42;
const file_header_size = 14;
const info_header_size = 40;

pub const Roi = struct {
    width: u32,
    height: u32,
};

pub fn Plane(comptime T: type) type {
    return struct {
        data: []T,
        stride: u32,

        pub fn free(plane: @This(), gpa: std.mem.Allocator) void {
            gpa.free(plane.data);
        }
    };
}

/// Clamps the input value to integer byte range [0, 255].
pub fn clamp0255(x: i32) i32 {
    return std.math.clamp(x, 0, 255);
}

pub fn roundF(num: f32) f32 {
    const num_abs = @abs(num);
    const num_abs_i: i32 = @intFromFloat(num_abs + 0.5);
    const sign: f32 = if (num > 0) 1.0 else -1.0;
    return sign * @as(f32, @floatFromInt(num_abs_i));
}

/// Memory allocator, returns aligned format frame with pixels of type `T`.
pub fn mallocPlane(comptime T: type, gpa: std.mem.Allocator, width: u32, height: u32) !Plane(T) {
    const stride_bytes = std.mem.alignForward(u32, width * @sizeOf(T), 16);
    const stride = stride_bytes / @sizeOf(T);
    return .{ .data = try gpa.alloc(T, stride * height), .stride = stride };
}

pub fn copyByte2Float(img_src: Plane(u8), img_dst: Plane(f32), size: Roi) void {
    for (0..size.height) |i| {
        for (0..size.width) |j| {
            const pixel = img_src.data[i * img_src.stride + j];
            img_dst.data[i * img_dst.stride + j] = @floatFromInt(pixel);
        }
    }
}

/// Copies float plane to byte plane (with clamp).
pub fn copyFloat2Byte(img_src: Plane(f32), img_dst: Plane(u8), size: Roi) void {
    for (0..size.height) |i| {
        for (0..size.width) |j| {
            const rounded = roundF(img_src.data[i * img_src.stride + j]);
            img_dst.data[i * img_dst.stride + j] = @intCast(clamp0255(@intFromFloat(rounded)));
        }
    }
}

pub fn addFloatPlane(value: f32, img_src_dst: Plane(f32), size: Roi) void {
    for (0..size.height) |i| {
        for (0..size.width) |j| {
            img_src_dst.data[i * img_src_dst.stride + j] += value;
        }
    }
}

/// Acquires the dimensions of the bitmap in `file`.
pub fn preLoadBmp(file: []const u8) !Roi {
    if (file.len < file_header_size + info_header_size) return error.InvalidImage;
    if (std.mem.readInt(u16, file[0..2], .little) != bmp_signature) return error.InvalidImage;
    const info = file[file_header_size..][0..info_header_size];
    const color_depth = std.mem.readInt(u16, info[14..16], .little);
    const compressed = std.mem.readInt(u32, info[16..20], .little);
    if (color_depth != 24 or compressed != 0) return error.InvalidImage;
    return .{
        .width = std.mem.readInt(u32, info[4..8], .little),
        .height = std.mem.readInt(u32, info[8..12], .little),
    };
}

/// Loads the luma of the bitmap in `file`.
pub fn loadBmpAsGray(file: []const u8, im_size: Roi, img: Plane(u8)) void {
    var pixels = file[file_header_size + info_header_size ..];
    var i = im_size.height;
    while (i > 0) {
        i -= 1;
        for (0..im_size.width) |j| {
            const b: i32 = pixels[0];
            const g: i32 = pixels[1];
            const r: i32 = pixels[2];
            pixels = pixels[3..];
            const val = (313524 * r + 615514 * g + 119537 * b + 524288) >> 20;
            img.data[i * img.stride + j] = @intCast(clamp0255(val));
        }
    }
}

/// Dumps the luma `img` as a bitmap to the file `file_name`.
pub fn dumpBmpAsGray(
    gpa: std.mem.Allocator,
    io: std.Io,
    file_name: []const u8,
    img: Plane(u8),
    im_size: Roi,
) !void {
    const pixel_bytes = 3 * im_size.width * im_size.height;
    const file = try gpa.alloc(u8, file_header_size + info_header_size + pixel_bytes);
    defer gpa.free(file);
    @memset(file[0 .. file_header_size + info_header_size], 0);

    std.mem.writeInt(u16, file[0..2], bmp_signature, .little);
    std.mem.writeInt(u32, file[2..6], @intCast(file.len), .little);
    std.mem.writeInt(u32, file[10..14], file_header_size + info_header_size, .little);
    const info = file[file_header_size..][0..info_header_size];
    std.mem.writeInt(u32, info[0..4], info_header_size, .little);
    std.mem.writeInt(u32, info[4..8], im_size.width, .little);
    std.mem.writeInt(u32, info[8..12], im_size.height, .little);
    std.mem.writeInt(u16, info[12..14], 1, .little);
    std.mem.writeInt(u16, info[14..16], 24, .little);

    var pixels = file[file_header_size + info_header_size ..];
    var i = im_size.height;
    while (i > 0) {
        i -= 1;
        for (0..im_size.width) |j| {
            @memset(pixels[0..3], img.data[i * img.stride + j]);
            pixels = pixels[3..];
        }
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file_name, .data = file });
}

/// Evaluates the Mean Square Error between two images.
pub fn calculateMse(img1: Plane(u8), img2: Plane(u8), size: Roi) f32 {
    var acc: u32 = 0;
    for (0..size.height) |i| {
        for (0..size.width) |j| {
            const pixel1: i32 = img1.data[i * img1.stride + j];
            const pixel2: i32 = img2.data[i * img2.stride + j];
            var tmp_diff = pixel1 - pixel2;
            tmp_diff *= tmp_diff;
            acc +%= @bitCast(tmp_diff);
        }
    }
    return @as(f32, @floatFromInt(acc)) / @as(f32, @floatFromInt(size.height * size.width));
}

/// Evaluates the Peak Signal to Noise Ratio between two images.
pub fn calculatePsnr(img1: Plane(u8), img2: Plane(u8), size: Roi) f32 {
    const mse = calculateMse(img1, img2, size);
    return 10 * @log10(255.0 * 255.0 / mse);
}
