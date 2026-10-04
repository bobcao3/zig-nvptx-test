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

//! This sample demonstrates how to use texture fetches from cubemap textures in CUDA.
//!
//! This sample first generates a cubemap input data array and the expected output. Then it starts
//! a kernel which fetches the texture data of each face (using the direction vector of each
//! texel), transforms it to the expected output, and writes it to a linear output array.

const std = @import("std");
const cuda = @import("cuda");

const sample_name = "simpleCubemapTexture";

const min_epsilon_error = 5e-3;

/// Returns whether every element differs by at most `epsilon`.
fn compareData(reference: []const f32, data: []const f32, epsilon: f32) bool {
    var result = true;
    for (reference, data) |expected, actual| {
        const diff = expected - actual;
        result = result and diff <= epsilon and diff >= -epsilon;
    }
    return result;
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
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("simpleCubemapTexture_kernel.ptx"));
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

    if (capability.major < 2) {
        std.log.warn("{s} requires SM 2.0 or higher for support of Texture Arrays.  Test will exit... ", .{sample_name});
        std.process.exit(2);
    }

    // Generate the input data for the cubemap.
    const width = 64;
    const num_faces = 6;
    const num_layers = 1;
    const cubemap_size = width * width * num_faces;
    const h_data = try gpa.alloc(f32, cubemap_size * num_layers);
    defer gpa.free(h_data);

    for (h_data, 0..) |*element, i| element.* = @floatFromInt(i);

    // This is the expected transformation of the input data (the expected output).
    const h_data_ref = try gpa.alloc(f32, h_data.len);
    defer gpa.free(h_data_ref);

    for (0..num_layers) |layer| {
        for (0..cubemap_size) |i| {
            h_data_ref[layer * cubemap_size + i] = -h_data[layer * cubemap_size + i] + @as(f32, @floatFromInt(layer));
        }
    }

    // Allocate device memory for the result.
    const d_data = try cuda.alloc(f32, h_data.len);
    defer d_data.free();

    // Allocate the array and copy the image data.
    const cu_3darray = try cuda.Array.create3d(.{
        .width = width,
        .height = width,
        .depth = num_faces,
        .format = .float,
        .channels = 1,
        .flags = .{ .cubemap = true },
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
        .grid = .{ .x = width / 8, .y = width / 8 },
        .block = .{ .x = 8, .y = 8 },
    };

    std.log.info("Covering Cubemap data array of {d}~3 x {d}: Grid size is {d} x {d}, each block has 8 x 8 threads", .{
        width,
        num_layers,
        config.grid.x,
        config.grid.y,
    });

    // Warmup (for better timing).
    try transform_kernel.launch(config, .{ d_data, @as(i32, width), tex });
    try cuda.synchronize();

    const start = std.Io.Clock.awake.now(init.io);

    // Execute the kernel.
    try transform_kernel.launch(config, .{ d_data, @as(i32, width), tex });

    try cuda.synchronize();
    const stop = std.Io.Clock.awake.now(init.io);
    const milliseconds = @as(f64, @floatFromInt(start.durationTo(stop).toNanoseconds())) * 1.0e-6;
    std.log.info("Processing time: {d:.3} msec", .{milliseconds});
    std.log.info("{d:.2} Mtexlookups/sec", .{@as(f64, cubemap_size) / (milliseconds / 1000.0) / 1e6});

    // Copy the result from the device to the host.
    const h_odata = try gpa.alloc(f32, h_data.len);
    defer gpa.free(h_odata);
    try d_data.copyTo(h_odata);

    std.log.info("Comparing kernel output to expected data", .{});
    if (!compareData(h_odata[0..cubemap_size], h_data_ref[0..cubemap_size], min_epsilon_error)) std.process.exit(1);
}
