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

//! This sample implements a separable convolution filter of a 2D image with an arbitrary kernel.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("convolutionSeparable_common.zig");

const kernel_radius = common.kernel_radius;
const kernel_length = common.kernel_length;

fn convolutionRowCpu(
    h_dst: []f32,
    h_src: []const f32,
    h_kernel: []const f32,
    image_w: usize,
    image_h: usize,
    kernel_r: usize,
) void {
    for (0..image_h) |y| {
        for (0..image_w) |x| {
            var sum: f32 = 0;
            for (0..2 * kernel_r + 1) |k| {
                if (x + k < kernel_r or x + k - kernel_r >= image_w) continue;
                sum += h_src[y * image_w + x + k - kernel_r] * h_kernel[2 * kernel_r - k];
            }
            h_dst[y * image_w + x] = sum;
        }
    }
}

fn convolutionColumnCpu(
    h_dst: []f32,
    h_src: []const f32,
    h_kernel: []const f32,
    image_w: usize,
    image_h: usize,
    kernel_r: usize,
) void {
    for (0..image_h) |y| {
        for (0..image_w) |x| {
            var sum: f32 = 0;
            for (0..2 * kernel_r + 1) |k| {
                if (y + k < kernel_r or y + k - kernel_r >= image_h) continue;
                sum += h_src[(y + k - kernel_r) * image_w + x] * h_kernel[2 * kernel_r - k];
            }
            h_dst[y * image_w + x] = sum;
        }
    }
}

const Gpu = struct {
    convolution_rows: cuda.Function,
    convolution_columns: cuda.Function,

    fn convolutionRows(
        gpu: Gpu,
        d_dst: cuda.Buffer(f32),
        d_src: cuda.Buffer(f32),
        image_w: u32,
        image_h: u32,
    ) !void {
        std.debug.assert(common.rows_blockdim_x * common.rows_halo_steps >= kernel_radius);
        std.debug.assert(image_w % (common.rows_result_steps * common.rows_blockdim_x) == 0);
        std.debug.assert(image_h % common.rows_blockdim_y == 0);

        const pitch: i32 = @intCast(image_w);
        try gpu.convolution_rows.launch(.{
            .grid = .{
                .x = image_w / (common.rows_result_steps * common.rows_blockdim_x),
                .y = image_h / common.rows_blockdim_y,
            },
            .block = .{ .x = common.rows_blockdim_x, .y = common.rows_blockdim_y },
        }, .{ d_dst, d_src, @as(i32, @intCast(image_w)), @as(i32, @intCast(image_h)), pitch });
    }

    fn convolutionColumns(
        gpu: Gpu,
        d_dst: cuda.Buffer(f32),
        d_src: cuda.Buffer(f32),
        image_w: u32,
        image_h: u32,
    ) !void {
        std.debug.assert(common.columns_blockdim_y * common.columns_halo_steps >= kernel_radius);
        std.debug.assert(image_w % common.columns_blockdim_x == 0);
        std.debug.assert(image_h % (common.columns_result_steps * common.columns_blockdim_y) == 0);

        const pitch: i32 = @intCast(image_w);
        try gpu.convolution_columns.launch(.{
            .grid = .{
                .x = image_w / common.columns_blockdim_x,
                .y = image_h / (common.columns_result_steps * common.columns_blockdim_y),
            },
            .block = .{ .x = common.columns_blockdim_x, .y = common.columns_blockdim_y },
        }, .{ d_dst, d_src, @as(i32, @intCast(image_w)), @as(i32, @intCast(image_h)), pitch });
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("[convolutionSeparable] - Starting...", .{});

    const image_w = 3072;
    const image_h = 3072;
    const iterations = 16;

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("convolutionSeparable_kernel.ptx"));
    defer module.unload();
    const gpu: Gpu = .{
        .convolution_rows = try module.function("convolutionRowsKernel"),
        .convolution_columns = try module.function("convolutionColumnsKernel"),
    };

    std.log.info("Image Width x Height = {d} x {d}", .{ image_w, image_h });
    std.log.info("Allocating and initializing host arrays...", .{});
    const h_kernel = try gpa.alloc(f32, kernel_length);
    defer gpa.free(h_kernel);
    const h_input = try gpa.alloc(f32, image_w * image_h);
    defer gpa.free(h_input);
    const h_buffer = try gpa.alloc(f32, image_w * image_h);
    defer gpa.free(h_buffer);
    const h_output_cpu = try gpa.alloc(f32, image_w * image_h);
    defer gpa.free(h_output_cpu);
    const h_output_gpu = try gpa.alloc(f32, image_w * image_h);
    defer gpa.free(h_output_gpu);

    var prng: std.Random.DefaultPrng = .init(200);
    const random = prng.random();
    for (h_kernel) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 16));
    for (h_input) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 16));

    std.log.info("Allocating and initializing CUDA arrays...", .{});
    const d_input = try cuda.alloc(f32, image_w * image_h);
    defer d_input.free();
    const d_output = try cuda.alloc(f32, image_w * image_h);
    defer d_output.free();
    const d_buffer = try cuda.alloc(f32, image_w * image_h);
    defer d_buffer.free();

    const c_kernel = try module.global(f32, "c_kernel");
    try c_kernel.copyFrom(h_kernel);
    try d_input.copyFrom(h_input);

    std.log.info("Running GPU convolution ({d} identical iterations)...", .{iterations});

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    // The first iteration is a warmup iteration.
    try gpu.convolutionRows(d_buffer, d_input, image_w, image_h);
    try gpu.convolutionColumns(d_output, d_buffer, image_w, image_h);
    try cuda.synchronize();
    try start.record(.default);
    for (0..iterations) |_| {
        try gpu.convolutionRows(d_buffer, d_input, image_w, image_h);
        try gpu.convolutionColumns(d_output, d_buffer, image_w, image_h);
    }
    try stop.record(.default);
    try stop.synchronize();
    const gpu_time = 0.001 * try start.elapsed(stop) / iterations;
    std.log.info("convolutionSeparable, Throughput = {d:.4} MPixels/sec, Time = {d:.5} s, " ++
        "Size = {d} Pixels, NumDevsUsed = {d}, Workgroup = {d}", .{
        1.0e-6 * @as(f64, image_w * image_h) / gpu_time,
        gpu_time,
        image_w * image_h,
        1,
        0,
    });

    std.log.info("Reading back GPU results...", .{});
    try d_output.copyTo(h_output_gpu);

    std.log.info("Checking the results...", .{});
    std.log.info(" ...running convolutionRowCPU()", .{});
    convolutionRowCpu(h_buffer, h_input, h_kernel, image_w, image_h, kernel_radius);

    std.log.info(" ...running convolutionColumnCPU()", .{});
    convolutionColumnCpu(h_output_cpu, h_buffer, h_kernel, image_w, image_h, kernel_radius);

    std.log.info(" ...comparing the results", .{});
    var sum: f64 = 0;
    var delta: f64 = 0;
    for (h_output_gpu, h_output_cpu) |gpu_value, cpu_value| {
        delta += (gpu_value - cpu_value) * (gpu_value - cpu_value);
        sum += cpu_value * cpu_value;
    }

    const l2_norm = @sqrt(delta / sum);
    std.log.info(" ...Relative L2 norm: {E}", .{l2_norm});
    std.log.info("Shutting down...", .{});

    if (l2_norm > 1e-6) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }
    std.log.info("Test passed", .{});
}
