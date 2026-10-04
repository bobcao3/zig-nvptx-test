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

//! This sample demonstrates how to use texture fetches from layered 2D textures in CUDA.
//!
//! This sample first generates a 3D input data array for the layered texture and the expected
//! output. Then it starts kernels, one for each layer, which fetch their layer's texture data
//! (using normalized texture coordinates), transform it to the expected output, and write it to
//! a 3D output data array.

const std = @import("std");
const cuda = @import("cuda");

const sample_name = "simpleLayeredTexture";

const min_epsilon_error = 5e-3;

/// Returns whether every element differs by at most `epsilon`, or, when `threshold` is not
/// zero, whether fewer than `threshold` of the elements differ by more.
fn compareData(reference: []const f32, data: []const f32, epsilon: f32, threshold: f32) bool {
    var result = true;
    var error_count: u32 = 0;
    for (reference, data) |expected, actual| {
        const diff = expected - actual;
        const matches = diff <= epsilon and diff >= -epsilon;
        result = result and matches;
        if (!matches) error_count += 1;
    }
    if (threshold == 0.0) return result;
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("[{s}] - Starting...", .{sample_name});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("simpleLayeredTexture_kernel.ptx"));
    defer module.unload();
    const transform_kernel = try module.function("transformKernel");

    // Get the number of multiprocessors of this GPU.
    var name_buffer: [256]u8 = undefined;
    const capability = try device.computeCapability();
    std.log.info("CUDA device [{s}] has {d} Multi-Processors SM {d}.{d}", .{
        try device.name(&name_buffer),
        try device.attribute(.multiprocessor_count),
        capability.major,
        capability.minor,
    });

    // Generate the input data for the layered texture.
    const width = 512;
    const height = 512;
    const num_layers = 5;
    const h_data = try gpa.alloc(f32, width * height * num_layers);
    defer gpa.free(h_data);

    for (0..num_layers) |layer| {
        for (0..width * height) |i| {
            h_data[layer * width * height + i] = @floatFromInt(i);
        }
    }

    // This is the expected transformation of the input data (the expected output).
    const h_data_ref = try gpa.alloc(f32, h_data.len);
    defer gpa.free(h_data_ref);

    for (0..num_layers) |layer| {
        for (0..width * height) |i| {
            h_data_ref[layer * width * height + i] = -h_data[layer * width * height + i] + @as(f32, @floatFromInt(layer));
        }
    }

    // Allocate device memory for the result.
    const d_data = try cuda.alloc(f32, h_data.len);
    defer d_data.free();

    // Allocate the array and copy the image data.
    const cu_3darray = try cuda.Array.create3d(.{
        .width = width,
        .height = height,
        .depth = num_layers,
        .format = .float,
        .channels = 1,
        .flags = .{ .layered = true },
    });
    defer cu_3darray.destroy();
    try cu_3darray.copyFromHost(f32, h_data);

    const tex = try cuda.TextureObject.create(.{ .array = cu_3darray }, .{
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

    std.log.info("Covering 2D data array of {d} x {d}: Grid size is {d} x {d}, each block has 8 x 8 threads", .{
        width,
        height,
        config.grid.x,
        config.grid.y,
    });

    // Warmup (for better timing).
    try transform_kernel.launch(config, .{ d_data, @as(i32, width), @as(i32, height), @as(i32, 0), tex });
    try cuda.synchronize();

    const start = std.Io.Clock.awake.now(init.io);

    // Execute the kernel.
    for (0..num_layers) |layer| {
        try transform_kernel.launch(config, .{ d_data, @as(i32, width), @as(i32, height), @as(i32, @intCast(layer)), tex });
    }

    try cuda.synchronize();
    const stop = std.Io.Clock.awake.now(init.io);
    const milliseconds = @as(f64, @floatFromInt(start.durationTo(stop).toNanoseconds())) * 1.0e-6;
    std.log.info("Processing time: {d:.3} msec", .{milliseconds});
    std.log.info("{d:.2} Mtexlookups/sec", .{
        @as(f64, width * height * num_layers) / (milliseconds / 1000.0) / 1e6,
    });

    // Copy the result from the device to the host.
    const h_odata = try gpa.alloc(f32, h_data.len);
    defer gpa.free(h_odata);
    try d_data.copyTo(h_odata);

    std.log.info("Comparing kernel output to expected data", .{});
    if (!compareData(h_odata, h_data_ref, min_epsilon_error, 0.0)) std.process.exit(1);
}
