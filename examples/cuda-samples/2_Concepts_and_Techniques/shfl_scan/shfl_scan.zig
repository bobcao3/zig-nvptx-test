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

//! Shuffle intrinsics CUDA Sample
//! This sample demonstrates the use of the shuffle intrinsic.
//! First, a simple example of a prefix sum using the shuffle to
//! perform a scan operation is provided.
//! Secondly, a more involved example of computing an integral image
//! using the shuffle intrinsic is provided, where the shuffle
//! scan operation and shuffle xor operations are used.

const std = @import("std");
const cuda = @import("cuda");

const exit_waived = 2;

fn divUp(dividend: u32, divisor: u32) u32 {
    return std.math.divCeil(u32, dividend, divisor) catch unreachable;
}

fn elapsedMilliseconds(io: std.Io, start: std.Io.Timestamp) f64 {
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io));
    return @as(f64, @floatFromInt(elapsed.toNanoseconds())) * 1.0e-6;
}

/// This function verifies the shuffle scan result, for the simple prefix sum case.
fn cpuVerify(io: std.Io, h_data: []i32, h_result: []const i32) bool {
    for (0..h_data.len - 1) |i| {
        h_data[i + 1] = h_data[i] +% h_data[i + 1];
    }

    var diff: i32 = 0;

    for (h_data, h_result) |expected, actual| {
        diff +%= expected -% actual;
    }

    std.log.info("CPU verify result diff (GPUvsCPU) = {d}", .{diff});
    const test_result = diff == 0;

    const start = std.Io.Clock.awake.now(io);

    for (0..100) |_| {
        for (0..h_data.len - 1) |i| {
            h_data[i + 1] = h_data[i] +% h_data[i + 1];
        }
    }

    const cput = elapsedMilliseconds(io, start);
    std.log.info("CPU sum (naive) took {d:.6} ms", .{cput / 100});
    return test_result;
}

/// This verifies the row scan result for synthetic data of all 1's.
fn verifyDataRowSums(h_image: []const u32, w: u32, h: u32) u32 {
    var diff: u32 = 0;

    for (0..h) |j| {
        for (0..w) |i| {
            const gold: i32 = @intCast(i + 1);
            diff +%= @abs(gold -% @as(i32, @bitCast(h_image[j * w + i])));
        }
    }

    return diff;
}

fn shuffleSimpleTest(
    gpa: std.mem.Allocator,
    io: std.Io,
    device: cuda.Device,
    module: cuda.Module,
) !bool {
    const n_elements = 65536;

    std.log.info("Starting shfl_scan", .{});

    const compute_capability = try device.computeCapability();
    const multiprocessor_count = try device.attribute(.multiprocessor_count);
    std.log.info("> Detected Compute SM {d}.{d} hardware with {d} multi-processors", .{
        compute_capability.major,
        compute_capability.minor,
        multiprocessor_count,
    });

    if (compute_capability.major < 3) {
        std.log.warn("> __shfl() intrinsic requires device SM 3.0+", .{});
        std.log.warn("> Waiving test.", .{});
        std.process.exit(exit_waived);
    }

    const h_data = try gpa.alloc(i32, n_elements);
    defer gpa.free(h_data);
    const h_result = try gpa.alloc(i32, n_elements);
    defer gpa.free(h_result);

    std.log.info("Computing Simple Sum test", .{});
    std.log.info("---------------------------------------------------", .{});

    std.log.info("Initialize test data [1, 1, 1...]", .{});

    @memset(h_data, 1);

    const block_size = 256;
    const grid_size = n_elements / block_size;
    const n_warps = block_size / 32;
    const shmem_sz = n_warps * @sizeOf(i32);
    const n_partial_sums = n_elements / block_size;

    std.log.info("Scan summation for {d} elements, {d} partial sums", .{
        n_elements,
        n_elements / block_size,
    });

    const p_block_size = @min(n_partial_sums, block_size);
    const p_grid_size = divUp(n_partial_sums, p_block_size);
    std.log.info("Partial summing {d} elements with {d} blocks of size {d}", .{
        n_partial_sums,
        p_grid_size,
        p_block_size,
    });

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    var et: f32 = 0;

    const d_data = try cuda.alloc(i32, n_elements);
    defer d_data.free();
    const d_partial_sums = try cuda.alloc(i32, n_partial_sums);
    defer d_partial_sums.free();
    try d_partial_sums.memset(0);

    const h_partial_sums = try gpa.alloc(i32, n_partial_sums);
    defer gpa.free(h_partial_sums);
    try d_data.copyFrom(h_data);

    const shfl_scan_test = try module.function("shflScanTest");
    const uniform_add = try module.function("uniformAdd");

    try start.record(.default);
    try shfl_scan_test.launch(.{
        .grid = .{ .x = grid_size },
        .block = .{ .x = block_size },
        .shared_memory_bytes = shmem_sz,
    }, .{ d_data, @as(u32, 32), d_partial_sums });
    try shfl_scan_test.launch(.{
        .grid = .{ .x = p_grid_size },
        .block = .{ .x = p_block_size },
        .shared_memory_bytes = shmem_sz,
    }, .{ d_partial_sums, @as(u32, 32), null });
    try uniform_add.launch(.{
        .grid = .{ .x = grid_size - 1 },
        .block = .{ .x = block_size },
    }, .{ d_data.address + block_size * @sizeOf(i32), d_partial_sums, @as(u32, n_elements) });
    try stop.record(.default);
    try stop.synchronize();
    et += try start.elapsed(stop);

    try d_data.copyTo(h_result);
    try d_partial_sums.copyTo(h_partial_sums);

    std.log.info("Test Sum: {d}", .{h_partial_sums[n_partial_sums - 1]});
    std.log.info("Time (ms): {d:.6}", .{et});
    std.log.info("{d} elements scanned in {d:.6} ms -> {d:.6} MegaElements/s", .{
        n_elements,
        et,
        n_elements / (et / 1000.0) / 1000000.0,
    });

    return cpuVerify(io, h_data, h_result);
}

/// This function tests creation of an integral image using synthetic data, of size 1920x1080
/// pixels greyscale.
fn shuffleIntegralImageTest(gpa: std.mem.Allocator, module: cuda.Module) !bool {
    const w = 1920;
    const h = 1080;
    const n_elements = w * h;

    std.log.info("Computing Integral Image Test on size {d} x {d} synthetic data", .{ w, h });
    std.log.info("---------------------------------------------------", .{});
    const h_image = try gpa.alloc(u32, n_elements);
    defer gpa.free(h_image);
    @memset(h_image, 0);

    // Each work item handles 16 values, with 1 work group per row.
    const block_size = divUp(w, 16);
    const grid_size = h;

    const d_data = try cuda.alloc(u8, n_elements * @sizeOf(u32));
    defer d_data.free();
    const d_integral_image = try cuda.alloc(u32, n_elements * 4);
    defer d_integral_image.free();
    try d_data.memset(1);
    try d_integral_image.memset(0);

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    var et: f32 = 0;

    const shfl_intimage_rows = try module.function("shflIntimageRows");
    const shfl_vertical_shfl = try module.function("shflVerticalShfl");

    try start.record(.default);
    try shfl_intimage_rows.launch(.{
        .grid = .{ .x = grid_size },
        .block = .{ .x = block_size },
    }, .{ d_data, d_integral_image });
    try stop.record(.default);
    try stop.synchronize();
    et = try start.elapsed(stop);
    std.log.info("Method: Fast  Time (GPU Timer): {d:.6} ms ", .{et});

    try d_integral_image.copyTo(h_image);
    const err = verifyDataRowSums(h_image, w, h);
    std.log.info("Diff = {d}", .{err});

    const block_sz: cuda.Dim3 = .{ .x = 32, .y = 8 };
    const test_grid: cuda.Dim3 = .{ .x = w / block_sz.x };

    try start.record(.default);
    try shfl_vertical_shfl.launch(.{
        .grid = test_grid,
        .block = block_sz,
    }, .{ d_integral_image, @as(u32, w), @as(u32, h) });
    try stop.record(.default);
    try stop.synchronize();
    et = try start.elapsed(stop);
    std.log.info("Method: Vertical Scan  Time (GPU Timer): {d:.6} ms ", .{et});

    try d_integral_image.copyTo(h_image);

    const final_sum = h_image[w * h - 1];
    std.log.info("CheckSum: {d}, (expect {d}x{d}={d})", .{ final_sum, w, h, w * h });

    // If the final value in the corner is the same as the size of the buffer (all 1's), then the
    // integral image was generated successfully.
    return final_sum == w * h;
}

pub fn main(init: std.process.Init) !void {
    std.log.info("Starting shfl_scan", .{});

    const device = try cuda.init(0);
    defer device.deinit();

    const compute_capability = try device.computeCapability();
    const multiprocessor_count = try device.attribute(.multiprocessor_count);
    std.log.info("> Detected Compute SM {d}.{d} hardware with {d} multi-processors", .{
        compute_capability.major,
        compute_capability.minor,
        multiprocessor_count,
    });

    if (compute_capability.major < 3) {
        std.log.warn("> __shfl() intrinsic requires device SM 3.0+", .{});
        std.log.warn("> Waiving test.", .{});
        std.process.exit(exit_waived);
    }

    const module = try cuda.Module.load(@embedFile("shfl_scan_kernel.ptx"));
    defer module.unload();

    const simple_test = try shuffleSimpleTest(init.gpa, init.io, device, module);
    const int_test = try shuffleIntegralImageTest(init.gpa, module);

    if (!(simple_test and int_test)) std.process.exit(1);
}
