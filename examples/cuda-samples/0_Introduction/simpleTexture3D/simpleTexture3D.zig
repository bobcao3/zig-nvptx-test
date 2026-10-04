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

//! 3D texture sample.
//!
//! This sample loads a 3D volume and renders a slice through it using 3D texture lookups. The
//! sample has no window: it renders one slice and compares it with the reference image.

const std = @import("std");
const cuda = @import("cuda");

const sample_name = "simpleTexture3D";

const max_epsilon_error = 5.0;
const threshold = 0.15;

const volume_filename = "Bucky.raw";
const volume_size: struct { width: u32, height: u32, depth: u32 } = .{ .width = 32, .height = 32, .depth = 32 };

const width = 512;
const height = 512;
const block_size: struct { x: u32, y: u32 } = .{ .x = 16, .y = 16 };
const grid_size: struct { x: u32, y: u32 } = .{ .x = width / block_size.x, .y = height / block_size.y };

/// The texture coordinate in z.
const w = 0.5;

const reference_filename = "ref_texture3D.bin";

/// Compares the output, reinterpreted as floats, with the reference file. Returns whether fewer
/// than `threshold` of the elements differ by `max_epsilon_error` or more.
fn compareBin2BinFloat(output: []const u32, reference: []const u8) bool {
    var error_count: u32 = 0;
    for (output, 0..) |value, i| {
        const expected: f32 = @bitCast(std.mem.readInt(u32, reference[i * 4 ..][0..4], .little));
        const actual: f32 = @bitCast(value);
        if (!(@abs(expected - actual) < max_epsilon_error)) error_count += 1;
    }
    std.log.info("> compareBin2Bin <float> nelements={d}, epsilon={d:.2}, threshold={d:.2}", .{
        output.len,
        max_epsilon_error,
        threshold,
    });
    if (error_count != 0) {
        std.log.info("{d:.2}(%) of bytes mismatched (count={d})", .{
            @as(f32, @floatFromInt(error_count)) * 100 / @as(f32, @floatFromInt(output.len)),
            error_count,
        });
    }
    const passed = @as(f32, @floatFromInt(output.len)) * threshold > @as(f32, @floatFromInt(error_count));
    if (passed) std.log.info("  OK", .{}) else std.log.err("  FAILURE: {d} errors...", .{error_count});
    return passed;
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
    std.log.info("{s} Starting...", .{sample_name});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("simpleTexture3D_kernel.ptx"));
    defer module.unload();
    const d_render = try module.function("dRender");

    // Load the volume data.
    const h_volume = @embedFile(volume_filename);
    std.log.info("Read '{s}', {d} bytes", .{ volume_filename, h_volume.len });

    // Create the 3D array and copy the data to it.
    const d_volume_array = try cuda.Array.create3d(.{
        .width = volume_size.width,
        .height = volume_size.height,
        .depth = volume_size.depth,
        .format = .unsigned_int8,
        .channels = 1,
    });
    defer d_volume_array.destroy();
    try d_volume_array.copyFromHost(u8, h_volume);

    const tex = try cuda.TextureObject.create(.{ .array = d_volume_array }, .{
        // Access with normalized texture coordinates.
        .normalized_coordinates = true,
        // Linear interpolation.
        .filter_mode = .linear,
        // Wrap the texture coordinates.
        .address_mode = .{ .wrap, .wrap, .wrap },
        .read_mode = .normalized_float,
    });
    defer tex.destroy();

    const d_output = try cuda.alloc(u32, width * height);
    defer d_output.free();

    // Render the volume data.
    try d_render.launch(.{
        .grid = .{ .x = grid_size.x, .y = grid_size.y },
        .block = .{ .x = block_size.x, .y = block_size.y },
    }, .{ d_output, @as(u32, width), @as(u32, height), @as(f32, w), tex });
    try cuda.synchronize();

    const h_output = try gpa.alloc(u32, width * height);
    defer gpa.free(h_output);
    try d_output.copyTo(h_output);

    if (!compareBin2BinFloat(h_output, @embedFile(reference_filename))) std.process.exit(1);
}
