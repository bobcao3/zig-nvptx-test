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

//! This sample demonstrates how use texture fetches in CUDA
//!
//! This sample takes an input PGM image (imageFilename) and generates
//! an output PGM image (imageFilename_out).  This CUDA kernel performs
//! a simple 2D transform (rotation) on the texture coordinates (u,v).

const std = @import("std");
const cuda = @import("cuda");

const min_epsilon_error = 5e-3;

const image_filename = "teapot512.pgm";
const reference_filename = "ref_rotated.pgm";
const angle: f32 = 0.5;

const sample_name = "simpleSurfaceWrite";

const Image = struct {
    width: u32,
    height: u32,
    pixels: []f32,
};

/// Parses a binary PGM file into pixels scaled to [0, 1].
fn loadPgm(gpa: std.mem.Allocator, file: []const u8) !Image {
    var tokens = std.mem.tokenizeAny(u8, file, " \t\r\n");
    if (!std.mem.eql(u8, tokens.next() orelse return error.InvalidImage, "P5")) {
        return error.InvalidImage;
    }
    var fields: [3]u32 = undefined;
    for (&fields) |*field| {
        var token = tokens.next() orelse return error.InvalidImage;
        while (token[0] == '#') {
            const comment_end = std.mem.indexOfScalarPos(u8, file, tokens.index, '\n') orelse
                return error.InvalidImage;
            tokens.index = comment_end;
            token = tokens.next() orelse return error.InvalidImage;
        }
        field.* = try std.fmt.parseInt(u32, token, 10);
    }
    const width, const height, _ = fields;
    const bytes = file[tokens.index + 1 ..][0 .. width * height];
    const pixels = try gpa.alloc(f32, bytes.len);
    for (pixels, bytes) |*pixel, byte| pixel.* = @as(f32, @floatFromInt(byte)) / 255.0;
    return .{ .width = width, .height = height, .pixels = pixels };
}

fn savePgm(gpa: std.mem.Allocator, io: std.Io, sub_path: []const u8, image: Image) !void {
    const header = try std.fmt.allocPrint(gpa, "P5\n{d}\n{d}\n255\n", .{ image.width, image.height });
    defer gpa.free(header);
    const file = try gpa.alloc(u8, header.len + image.pixels.len);
    defer gpa.free(file);
    @memcpy(file[0..header.len], header);
    for (file[header.len..], image.pixels) |*byte, pixel| byte.* = @intFromFloat(pixel * 255.0);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sub_path, .data = file });
}

/// Returns whether all elements of `data` are within `epsilon` of the elements of `reference`.
fn compareData(reference: []const f32, data: []const f32, epsilon: f32) bool {
    var result = true;
    for (reference, data) |expected, actual| {
        const diff = expected - actual;
        result = result and diff <= epsilon and diff >= -epsilon;
    }
    return result;
}

fn runTest(init: std.process.Init) !bool {
    const gpa = init.gpa;
    const device = try cuda.init(0);
    defer device.deinit();

    var name_buffer: [256]u8 = undefined;
    const capability = try device.computeCapability();
    std.log.info("CUDA device [{s}] has {d} Multi-Processors, SM {d}.{d}", .{
        try device.name(&name_buffer),
        try device.attribute(.multiprocessor_count),
        capability.major,
        capability.minor,
    });

    const image = try loadPgm(gpa, @embedFile(image_filename));
    defer gpa.free(image.pixels);
    const width = image.width;
    const height = image.height;
    std.log.info("Loaded '{s}', {d} x {d} pixels", .{ image_filename, width, height });

    const reference = try loadPgm(gpa, @embedFile(reference_filename));
    defer gpa.free(reference.pixels);

    const module = try cuda.Module.load(@embedFile("simpleSurfaceWrite_kernel.ptx"));
    defer module.unload();
    const surface_write_kernel = try module.function("surfaceWriteKernel");
    const transform_kernel = try module.function("transformKernel");

    const d_data = try cuda.alloc(f32, width * height);
    defer d_data.free();

    const array = try cuda.Array.create3d(.{
        .width = width,
        .height = height,
        .format = .float,
        .channels = 1,
        .flags = .{ .surface_ldst = true },
    });
    defer array.destroy();

    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = width / 8, .y = height / 8 },
        .block = .{ .x = 8, .y = 8 },
    };

    const output_surface = try cuda.SurfaceObject.create(.{ .array = array });
    defer output_surface.destroy();
    try d_data.copyFrom(image.pixels);
    try surface_write_kernel.launch(config, .{ d_data, @as(i32, @intCast(width)), @as(i32, @intCast(height)), output_surface });

    const tex = try cuda.TextureObject.create(.{ .array = array }, .{
        .normalized_coordinates = true,
        .filter_mode = .linear,
        .address_mode = .{ .wrap, .wrap, .clamp },
        .read_mode = .element_type,
    });
    defer tex.destroy();
    const transform_args = .{ d_data, @as(i32, @intCast(width)), @as(i32, @intCast(height)), angle, tex };

    // Warmup
    try transform_kernel.launch(config, transform_args);
    try cuda.synchronize();

    const start = std.Io.Clock.awake.now(init.io);
    try transform_kernel.launch(config, transform_args);
    try cuda.synchronize();
    const elapsed_ns: f32 = @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds());
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;
    std.log.info("Processing time: {d:.6} (ms)", .{elapsed_ms});
    std.log.info("{d:.2} Mpixels/sec", .{
        (@as(f32, @floatFromInt(width * height)) / (elapsed_ms / 1000.0)) / 1e6,
    });

    const output = try gpa.alloc(f32, width * height);
    defer gpa.free(output);
    try d_data.copyTo(output);

    const output_filename = "output.pgm";
    var result: Image = .{ .width = width, .height = height, .pixels = output };
    try savePgm(gpa, init.io, output_filename, result);
    std.log.info("Wrote '{s}'", .{output_filename});

    // The file stores 8-bit pixels, so the comparison reloads the data as the original does.
    const file = try std.Io.Dir.cwd().readFileAlloc(init.io, output_filename, gpa, .unlimited);
    defer gpa.free(file);
    result = try loadPgm(gpa, file);
    defer gpa.free(result.pixels);

    std.log.info("Comparing files", .{});
    std.log.info("\toutput:    <{s}>", .{output_filename});
    std.log.info("\treference: <{s}>", .{reference_filename});
    return compareData(result.pixels, reference.pixels, min_epsilon_error);
}

pub fn main(init: std.process.Init) !void {
    std.log.info("{s} starting...", .{sample_name});
    const test_result = try runTest(init);
    std.log.info("{s} completed, returned {s}", .{ sample_name, if (test_result) "OK" else "ERROR!" });
    if (!test_result) std.process.exit(1);
}
