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

//! This sample demonstrates how use texture fetches in CUDA.
//!
//! This sample takes an input PGM image (image_filename) and generates an output PGM image
//! (image_filename_out). This CUDA kernel performs a simple 2D transform (rotation) on the
//! texture coordinates (u,v).

const std = @import("std");
const cuda = @import("cuda");
const pgm = @import("pgm.zig");

const max_epsilon_error = 5e-3;

const image_filename = "teapot512.pgm";
const reference_filename = "ref_rotated.pgm";
const sample_name = "simpleTexture";

/// The angle to rotate the image by, in radians.
const angle = 0.5;

/// Returns whether fewer than `threshold` of the elements differ by more than `epsilon`.
fn compareData(reference: []const f32, data: []const f32, epsilon: f32, threshold: f32) bool {
    var error_count: u32 = 0;
    for (reference, data) |expected, actual| {
        const diff = expected - actual;
        if (!(diff <= epsilon and diff >= -epsilon)) error_count += 1;
    }
    if (error_count != 0) {
        std.log.info("{d:.2}(%) of bytes mismatched (count={d})", .{
            @as(f32, @floatFromInt(error_count)) * 100 / @as(f32, @floatFromInt(reference.len)),
            error_count,
        });
    }
    return @as(f32, @floatFromInt(reference.len)) * threshold > @as(f32, @floatFromInt(error_count));
}

fn flagValue(args: []const [:0]const u8, name: []const u8) !?u32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') return try std.fmt.parseInt(u32, flag[name.len + 1 ..], 10);
    }
    return null;
}

/// Runs a simple test for CUDA and returns whether the result matches the reference image.
fn runTest(init: std.process.Init, args: []const [:0]const u8) !bool {
    const gpa = init.gpa;
    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("simpleTexture_kernel.ptx"));
    defer module.unload();
    const transform_kernel = try module.function("transformKernel");

    // Load the image from the embedded file.
    const image = try pgm.load(gpa, @embedFile(image_filename));
    defer gpa.free(image.pixels);
    const width = image.width;
    const height = image.height;
    std.log.info("Loaded '{s}', {d} x {d} pixels", .{ image_filename, width, height });

    // Load the reference image (output).
    const reference = try pgm.load(gpa, @embedFile(reference_filename));
    defer gpa.free(reference.pixels);

    // Allocate device memory for the result.
    const d_data = try cuda.alloc(f32, width * height);
    defer d_data.free();

    // Allocate an array and copy the image data.
    const cu_array = try cuda.Array.create(.{
        .width = width,
        .height = height,
        .format = .float,
        .channels = 1,
    });
    defer cu_array.destroy();
    try cu_array.copyFromHost(f32, image.pixels);

    const tex = try cuda.TextureObject.create(.{ .array = cu_array }, .{
        .normalized_coordinates = true,
        .filter_mode = .linear,
        .address_mode = .{ .wrap, .wrap, .wrap },
        .read_mode = .element_type,
    });
    defer tex.destroy();

    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = width / 8, .y = height / 8 },
        .block = .{ .x = 8, .y = 8 },
    };
    const kernel_args = .{ d_data, @as(i32, @intCast(width)), @as(i32, @intCast(height)), @as(f32, angle), tex };

    // Warmup.
    try transform_kernel.launch(config, kernel_args);
    try cuda.synchronize();
    const start = std.Io.Clock.awake.now(init.io);

    // Execute the kernel.
    try transform_kernel.launch(config, kernel_args);
    try cuda.synchronize();
    const stop = std.Io.Clock.awake.now(init.io);
    const milliseconds = @as(f64, @floatFromInt(start.durationTo(stop).toNanoseconds())) * 1.0e-6;
    std.log.info("Processing time: {d:.6} (ms)", .{milliseconds});
    std.log.info("{d:.2} Mpixels/sec", .{
        (@as(f64, @floatFromInt(width * height)) / (milliseconds / 1000.0)) / 1e6,
    });

    // Copy the result from the device to the host.
    const h_output_data = try gpa.alloc(f32, width * height);
    defer gpa.free(h_output_data);
    try d_data.copyTo(h_output_data);

    // Write the result to a file.
    const output_filename = image_filename[0 .. image_filename.len - ".pgm".len] ++ "_out.pgm";
    const output_file = try pgm.save(gpa, .{ .width = width, .height = height, .pixels = h_output_data });
    defer gpa.free(output_file);
    const cwd = std.Io.Dir.cwd();
    try cwd.writeFile(init.io, .{ .sub_path = output_filename, .data = output_file });
    defer cwd.deleteFile(init.io, output_filename) catch {};
    std.log.info("Wrote '{s}'", .{output_filename});

    // Reload the data from the file, because saving quantizes it to bytes.
    const written = try cwd.readFileAlloc(init.io, output_filename, gpa, .unlimited);
    defer gpa.free(written);
    const output = try pgm.load(gpa, written);
    defer gpa.free(output.pixels);

    std.log.info("Comparing files", .{});
    std.log.info("\toutput:    <{s}>", .{output_filename});
    std.log.info("\treference: <{s}>", .{reference_filename});
    return compareData(output.pixels, reference.pixels, max_epsilon_error, 0.15);
}

pub fn main(init: std.process.Init) !void {
    std.log.info("{s} starting...", .{sample_name});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const test_result = try runTest(init, args);

    std.log.info("{s} completed, returned {s}", .{ sample_name, if (test_result) "OK" else "ERROR!" });
    if (!test_result) std.process.exit(1);
}
