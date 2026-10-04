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

//! This example demonstrates how to use textures bound to pitch linear memory. It performs a
//! shift of matrix elements using wrap addressing mode (aka periodic boundary conditions) on two
//! arrays, a pitch linear and a CUDA array, in order to highlight the differences in using each.
//!
//! Textures binding to pitch linear memory is a new feature in CUDA 2.2, and allows use of
//! texture features such as wrap addressing mode and filtering which are not possible with
//! textures bound to regular linear memory.

const std = @import("std");
const cuda = @import("cuda");

/// The number of repetitions performed.
const num_reps = 100;
/// The tile and block size.
const tile_dim = 16;

const sample_name = "simplePitchLinearTexture";

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

const Pitched = struct {
    buffer: cuda.Buffer(f32),
    pitch_bytes: usize,
};

/// Allocates device memory for `height` rows whose pitch meets the alignment that textures
/// require. The copies below treat the memory as dense, so the pitch must equal the row size.
fn allocPitched(device: cuda.Device, width_bytes: usize, height: usize) !Pitched {
    const alignment: usize = @intCast(try device.attribute(.texture_pitch_alignment));
    const pitch_bytes = std.mem.alignForward(usize, width_bytes, alignment);
    std.debug.assert(pitch_bytes == width_bytes);
    return .{ .buffer = try cuda.alloc(f32, pitch_bytes * height / @sizeOf(f32)), .pitch_bytes = pitch_bytes };
}

/// Runs a simple test for CUDA and returns whether both kernels match the reference.
fn runTest(init: std.process.Init, args: []const [:0]const u8) !bool {
    const gpa = init.gpa;

    // Set the array size.
    const nx = 2048;
    const ny = 2048;

    // Set up the shifts applied to the x and y data.
    const x_shift = 5;
    const y_shift = 7;

    if (nx % tile_dim != 0 or ny % tile_dim != 0) {
        std.log.err("nx and ny must be multiples of tile_dim", .{});
        std.process.exit(1);
    }

    // Set up the execution configuration parameters.
    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = nx / tile_dim, .y = ny / tile_dim },
        .block = .{ .x = tile_dim, .y = tile_dim },
    };

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("simplePitchLinearTexture_kernel.ptx"));
    defer module.unload();
    const shift_pitch_linear = try module.function("shiftPitchLinear");
    const shift_array = try module.function("shiftArray");

    // CUDA events for timing.
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    // Host allocation and initialization.
    const h_idata = try gpa.alloc(f32, nx * ny);
    defer gpa.free(h_idata);
    const h_odata = try gpa.alloc(f32, nx * ny);
    defer gpa.free(h_odata);
    const gold = try gpa.alloc(f32, nx * ny);
    defer gpa.free(gold);

    for (h_idata, 0..) |*element, i| element.* = @floatFromInt(i);

    // Device memory allocation: the pitch linear input data.
    const d_idata_pl = try allocPitched(device, nx * @sizeOf(f32), ny);
    defer d_idata_pl.buffer.free();

    // The array input data.
    const d_idata_array = try cuda.Array.create(.{
        .width = nx,
        .height = ny,
        .format = .float,
        .channels = 1,
    });
    defer d_idata_array.destroy();

    // The pitch linear output data.
    const d_odata = try allocPitched(device, nx * @sizeOf(f32), ny);
    defer d_odata.buffer.free();

    // Copy the host data to the device: pitch linear first, then the array.
    const h_pitch_bytes = nx * @sizeOf(f32);
    try d_idata_pl.buffer.copyFrom(h_idata);
    try d_idata_array.copyFromHost(f32, h_idata);

    const tex_ref_pl = try cuda.TextureObject.create(.{ .pitch2d = .{
        .address = d_idata_pl.buffer.address,
        .format = .float,
        .channels = 1,
        .width = nx,
        .height = ny,
        .pitch_bytes = h_pitch_bytes,
    } }, .{
        .normalized_coordinates = true,
        .filter_mode = .point,
        .address_mode = .{ .wrap, .wrap, .wrap },
        .read_mode = .element_type,
    });
    defer tex_ref_pl.destroy();
    const tex_ref_array = try cuda.TextureObject.create(.{ .array = d_idata_array }, .{
        .normalized_coordinates = true,
        .filter_mode = .point,
        .address_mode = .{ .wrap, .wrap, .wrap },
        .read_mode = .element_type,
    });
    defer tex_ref_array.destroy();

    // Reference calculation.
    for (0..ny) |j| {
        const jshift = (j + y_shift) % ny;
        for (0..nx) |i| {
            const ishift = (i + x_shift) % nx;
            gold[j * nx + i] = h_idata[jshift * nx + ishift];
        }
    }

    const pitch: i32 = @intCast(d_odata.pitch_bytes / @sizeOf(f32));

    // Run the shiftPitchLinear kernel.
    try d_odata.buffer.memset(0);
    try start.record(.default);
    for (0..num_reps) |_| {
        try shift_pitch_linear.launch(config, .{ d_odata.buffer, pitch, @as(i32, nx), @as(i32, ny), @as(i32, x_shift), @as(i32, y_shift), tex_ref_pl });
    }
    try stop.record(.default);
    try stop.synchronize();
    const time_pl = try start.elapsed(stop);

    // Check the results.
    try d_odata.buffer.copyTo(h_odata);
    var test_result = true;
    if (!compareData(gold, h_odata, 0.0, 0.15)) {
        std.log.err("*** shiftPitchLinear failed ***", .{});
        test_result = false;
    }

    // Run the shiftArray kernel.
    try d_odata.buffer.memset(0);
    try start.record(.default);
    for (0..num_reps) |_| {
        try shift_array.launch(config, .{ d_odata.buffer, pitch, @as(i32, nx), @as(i32, ny), @as(i32, x_shift), @as(i32, y_shift), tex_ref_array });
    }
    try stop.record(.default);
    try stop.synchronize();
    const time_array = try start.elapsed(stop);

    // Check the results.
    try d_odata.buffer.copyTo(h_odata);
    if (!compareData(gold, h_odata, 0.0, 0.15)) {
        std.log.err("*** shiftArray failed ***", .{});
        test_result = false;
    }

    const bytes: f32 = 2.0 * 1000.0 * nx * ny * @sizeOf(f32) / 1.0e+9;
    const bandwidth_pl = bytes / (time_pl / num_reps);
    const bandwidth_array = bytes / (time_array / num_reps);
    std.log.info("Bandwidth (GB/s) for pitch linear: {e:.2}; for array: {e:.2}", .{ bandwidth_pl, bandwidth_array });

    const pixels: f32 = nx * ny;
    const fetch_rate_pl = pixels / 1.0e+6 / (time_pl / (1000.0 * num_reps));
    const fetch_rate_array = pixels / 1.0e+6 / (time_array / (1000.0 * num_reps));
    std.log.info("Texture fetch rate (Mpix/s) for pitch linear: {e:.2}; for array: {e:.2}", .{
        fetch_rate_pl,
        fetch_rate_array,
    });

    return test_result;
}

pub fn main(init: std.process.Init) !void {
    std.log.info("{s} starting...", .{sample_name});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const test_result = try runTest(init, args);

    std.log.info("{s} completed, returned {s}", .{ sample_name, if (test_result) "OK" else "ERROR!" });
    if (!test_result) std.process.exit(1);
}
