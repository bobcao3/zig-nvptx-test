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

//! This sample implements the same algorithm as the convolutionSeparable
//! CUDA Sample, but without using the shared memory at all.
//! Instead, it uses textures in exactly the same way an OpenGL-based
//! implementation would do.
//! Refer to the "Performance" section of convolutionSeparable whitepaper.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("convolutionTexture_common.zig");

const kernel_radius = common.kernel_radius;
const kernel_length = common.kernel_length;

fn iDivUp(a: u32, b: u32) u32 {
    return std.math.divCeil(u32, a, b) catch unreachable;
}

fn convolutionRowsCpu(
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
                const d = @min(x + k -| kernel_r, image_w - 1);
                sum += h_src[y * image_w + d] * h_kernel[2 * kernel_r - k];
            }
            h_dst[y * image_w + x] = sum;
        }
    }
}

fn convolutionColumnsCpu(
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
                const d = @min(y + k -| kernel_r, image_h - 1);
                sum += h_src[d * image_w + x] * h_kernel[2 * kernel_r - k];
            }
            h_dst[y * image_w + x] = sum;
        }
    }
}

fn convolutionRowsGpu(
    function: cuda.Function,
    d_dst: cuda.Buffer(f32),
    image_w: u32,
    image_h: u32,
    tex_src: cuda.TextureObject,
) !void {
    const threads: cuda.Dim3 = .{ .x = 16, .y = 12 };
    try function.launch(.{
        .grid = .{ .x = iDivUp(image_w, threads.x), .y = iDivUp(image_h, threads.y) },
        .block = threads,
    }, .{ d_dst, @as(i32, @intCast(image_w)), @as(i32, @intCast(image_h)), tex_src });
}

fn convolutionColumnsGpu(
    function: cuda.Function,
    d_dst: cuda.Buffer(f32),
    image_w: u32,
    image_h: u32,
    tex_src: cuda.TextureObject,
) !void {
    const threads: cuda.Dim3 = .{ .x = 16, .y = 12 };
    try function.launch(.{
        .grid = .{ .x = iDivUp(image_w, threads.x), .y = iDivUp(image_h, threads.y) },
        .block = threads,
    }, .{ d_dst, @as(i32, @intCast(image_w)), @as(i32, @intCast(image_h)), tex_src });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const image_w = 3072;
    const image_h = 3072 / 2;
    const iterations = 10;

    std.log.info("[{s}] - Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("convolutionTexture_kernel.ptx"));
    defer module.unload();
    const convolution_rows = try module.function("convolutionRowsKernel");
    const convolution_columns = try module.function("convolutionColumnsKernel");

    std.log.info("Initializing data...", .{});
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

    const a_src = try cuda.Array.create(.{
        .width = image_w,
        .height = image_h,
        .format = .float,
        .channels = 1,
    });
    defer a_src.destroy();
    const d_output = try cuda.alloc(f32, image_w * image_h);
    defer d_output.free();

    const tex_src = try cuda.TextureObject.create(.{ .array = a_src }, .{
        .normalized_coordinates = false,
        .filter_mode = .linear,
        .address_mode = .{ .wrap, .wrap, .clamp },
        .read_mode = .element_type,
    });
    defer tex_src.destroy();

    var prng: std.Random.DefaultPrng = .init(2009);
    const random = prng.random();
    for (h_kernel) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 16));
    for (h_input) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 16));

    const c_kernel = try module.global(f32, "c_kernel");
    try c_kernel.copyFrom(h_kernel);
    try a_src.copyFromHost(f32, h_input);

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    std.log.info("Running GPU rows convolution ({d} identical iterations)...", .{iterations});
    try cuda.synchronize();
    try start.record(.default);
    for (0..iterations) |_| {
        try convolutionRowsGpu(convolution_rows, d_output, image_w, image_h, tex_src);
    }
    try stop.record(.default);
    try stop.synchronize();
    var gpu_time = try start.elapsed(stop) / iterations;
    std.log.info("Average convolutionRowsGPU() time: {d:.6} msecs; //{d:.6} Mpix/s", .{
        gpu_time,
        image_w * image_h * 1e-6 / (0.001 * gpu_time),
    });

    // While CUDA kernels can't write to textures directly, this copy is
    // inevitable
    std.log.info("Copying convolutionRowGPU() output back to the texture...", .{});
    try cuda.synchronize();
    try start.record(.default);
    try a_src.copyFromBuffer(f32, d_output, image_w * @sizeOf(f32));
    try stop.record(.default);
    try stop.synchronize();
    gpu_time = try start.elapsed(stop);
    std.log.info("cudaMemcpyToArray() time: {d:.6} msecs; //{d:.6} Mpix/s", .{
        gpu_time,
        image_w * image_h * 1e-6 / (0.001 * gpu_time),
    });

    std.log.info("Running GPU columns convolution ({d} iterations)", .{iterations});
    try cuda.synchronize();
    try start.record(.default);
    for (0..iterations) |_| {
        try convolutionColumnsGpu(convolution_columns, d_output, image_w, image_h, tex_src);
    }
    try stop.record(.default);
    try stop.synchronize();
    gpu_time = try start.elapsed(stop) / iterations;
    std.log.info("Average convolutionColumnsGPU() time: {d:.6} msecs; //{d:.6} Mpix/s", .{
        gpu_time,
        image_w * image_h * 1e-6 / (0.001 * gpu_time),
    });

    std.log.info("Reading back GPU results...", .{});
    try d_output.copyTo(h_output_gpu);

    std.log.info("Checking the results...", .{});
    std.log.info("...running convolutionRowsCPU()", .{});
    convolutionRowsCpu(h_buffer, h_input, h_kernel, image_w, image_h, kernel_radius);

    std.log.info("...running convolutionColumnsCPU()", .{});
    convolutionColumnsCpu(h_output_cpu, h_buffer, h_kernel, image_w, image_h, kernel_radius);

    var delta: f64 = 0;
    var sum: f64 = 0;
    for (h_output_cpu, h_output_gpu) |cpu, gpu| {
        sum += cpu * cpu;
        delta += (gpu - cpu) * (gpu - cpu);
    }

    const l2_norm = @sqrt(delta / sum);
    std.log.info("Relative L2 norm: {e}", .{l2_norm});
    std.log.info("Shutting down...", .{});

    if (l2_norm > 1e-6) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }

    std.log.info("Test passed", .{});
}
