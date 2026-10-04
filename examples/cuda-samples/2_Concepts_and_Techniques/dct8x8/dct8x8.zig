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

//! Contains entry point, wrappers to host and device code and benchmark.
//!
//! This sample implements forward and inverse Discrete Cosine Transform to blocks of image pixels
//! (of 8x8 size), as in JPEG standard. The typical work flow is as follows:
//! 1. Run CPU version (Host code) and measure execution time;
//! 2. Run CUDA version (Device code) and measure execution time;
//! 3. Output execution timings and calculate CUDA speedup.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("common.zig");
const bmp_util = @import("bmp_util.zig");
const gold = @import("dct8x8_gold.zig");

const Roi = bmp_util.Roi;
const Plane = bmp_util.Plane;
const block_size = common.block_size;

/// The number of DCT kernel calls.
const benchmark_size = 10;

/// The PSNR values over this threshold indicate images equality.
const psnr_threshold_equal = 40;

const Io = std.Io;

const Timer = struct {
    io: Io,
    start_time: Io.Timestamp = undefined,
    total_ms: f32 = 0,
    runs: u32 = 0,

    fn start(timer: *Timer) void {
        timer.start_time = Io.Clock.awake.now(timer.io);
    }

    fn stop(timer: *Timer) void {
        const elapsed = timer.start_time.durationTo(Io.Clock.awake.now(timer.io));
        const elapsed_ns: f32 = @floatFromInt(elapsed.toNanoseconds());
        timer.total_ms += elapsed_ns / std.time.ns_per_ms;
        timer.runs += 1;
    }

    fn reset(timer: *Timer) void {
        timer.total_ms = 0;
        timer.runs = 0;
    }

    fn averageValue(timer: Timer) f32 {
        return timer.total_ms / @as(f32, @floatFromInt(timer.runs));
    }
};

/// Wrapper function for 1st gold version of DCT, quantization and IDCT implementations.
/// Returns execution time in milliseconds.
fn wrapperGold1(
    gpa: std.mem.Allocator,
    io: Io,
    img_src: Plane(u8),
    img_dst: Plane(u8),
    size: Roi,
) !f32 {
    const img_f1 = try bmp_util.mallocPlane(f32, gpa, size.width, size.height);
    defer img_f1.free(gpa);
    const img_f2 = try bmp_util.mallocPlane(f32, gpa, size.width, size.height);
    defer img_f2.free(gpa);
    const stride_f = img_f1.stride;

    bmp_util.copyByte2Float(img_src, img_f1, size);
    bmp_util.addFloatPlane(-128.0, img_f1, size);

    var timer_gold: Timer = .{ .io = io };
    for (0..benchmark_size) |_| {
        timer_gold.start();
        gold.computeDct8x8Gold1(img_f1.data, img_f2.data, stride_f, size);
        timer_gold.stop();
    }
    const timer_gold_span = timer_gold.averageValue();

    gold.quantizeGoldFloat(img_f2.data, stride_f, size);

    gold.computeIdct8x8Gold1(img_f2.data, img_f1.data, stride_f, size);

    bmp_util.addFloatPlane(128.0, img_f1, size);
    bmp_util.copyFloat2Byte(img_f1, img_dst, size);

    return timer_gold_span;
}

/// Wrapper function for 2nd gold version of DCT, quantization and IDCT implementations.
/// Returns execution time in milliseconds.
fn wrapperGold2(
    gpa: std.mem.Allocator,
    io: Io,
    img_src: Plane(u8),
    img_dst: Plane(u8),
    size: Roi,
) !f32 {
    const img_f1 = try bmp_util.mallocPlane(f32, gpa, size.width, size.height);
    defer img_f1.free(gpa);
    const img_f2 = try bmp_util.mallocPlane(f32, gpa, size.width, size.height);
    defer img_f2.free(gpa);
    const stride_f = img_f1.stride;

    bmp_util.copyByte2Float(img_src, img_f1, size);
    bmp_util.addFloatPlane(-128.0, img_f1, size);

    var timer_gold: Timer = .{ .io = io };
    for (0..benchmark_size) |_| {
        timer_gold.start();
        gold.computeDct8x8Gold2(img_f1.data, img_f2.data, stride_f, size);
        timer_gold.stop();
    }
    const timer_gold_span = timer_gold.averageValue();

    gold.quantizeGoldFloat(img_f2.data, stride_f, size);

    gold.computeIdct8x8Gold2(img_f2.data, img_f1.data, stride_f, size);

    bmp_util.addFloatPlane(128.0, img_f1, size);
    bmp_util.copyFloat2Byte(img_f1, img_dst, size);

    return timer_gold_span;
}

/// Wrapper function for 1st CUDA version of DCT, quantization and IDCT implementations.
/// Returns execution time in milliseconds.
fn wrapperCuda1(
    gpa: std.mem.Allocator,
    io: Io,
    module: cuda.Module,
    img_src: Plane(u8),
    img_dst: Plane(u8),
    size: Roi,
) !f32 {
    const kernel1_dct = try module.function("cudaKernel1Dct");
    const kernel1_idct = try module.function("cudaKernel1Idct");
    const quantization_float = try module.function("cudaKernelQuantizationFloat");

    const src = try cuda.Array.create(.{
        .width = size.width,
        .height = size.height,
        .format = .float,
        .channels = 1,
    });
    defer src.destroy();
    const img_src_f = try bmp_util.mallocPlane(f32, gpa, size.width, size.height);
    defer img_src_f.free(gpa);
    const dst_stride = img_src_f.stride;
    std.debug.assert(dst_stride == size.width);
    const dst = try cuda.alloc(f32, img_src_f.data.len);
    defer dst.free();

    bmp_util.copyByte2Float(img_src, img_src_f, size);
    bmp_util.addFloatPlane(-128.0, img_src_f, size);

    try src.copyFromHost(f32, img_src_f.data);

    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = size.width / block_size, .y = size.height / block_size },
        .block = .{ .x = block_size, .y = block_size },
    };

    var timer_cuda: Timer = .{ .io = io };

    const tex_src = try cuda.TextureObject.create(.{ .array = src }, .{
        .normalized_coordinates = false,
        .filter_mode = .linear,
        .address_mode = .{ .wrap, .wrap, .clamp },
        .read_mode = .element_type,
    });
    defer tex_src.destroy();

    for (0..benchmark_size) |_| {
        timer_cuda.start();
        try kernel1_dct.launch(config, .{ dst, dst_stride, @as(u32, 0), @as(u32, 0), tex_src });
        try cuda.synchronize();
        timer_cuda.stop();
    }

    const timer_cuda_span = timer_cuda.averageValue();

    try quantization_float.launch(config, .{ dst, dst_stride });

    try src.copyFromBuffer(f32, dst, dst_stride * @sizeOf(f32));

    try kernel1_idct.launch(config, .{ dst, dst_stride, @as(u32, 0), @as(u32, 0), tex_src });

    try dst.copyTo(img_src_f.data);

    bmp_util.addFloatPlane(128.0, img_src_f, size);
    bmp_util.copyFloat2Byte(img_src_f, img_dst, size);

    return timer_cuda_span;
}

/// Wrapper function for 2nd CUDA version of DCT, quantization and IDCT implementations.
/// Returns execution time in milliseconds.
fn wrapperCuda2(
    gpa: std.mem.Allocator,
    io: Io,
    module: cuda.Module,
    img_src: Plane(u8),
    img_dst: Plane(u8),
    size: Roi,
) !f32 {
    const kernel2_dct = try module.function("cudaKernel2Dct");
    const kernel2_idct = try module.function("cudaKernel2Idct");
    const quantization_float = try module.function("cudaKernelQuantizationFloat");

    const img_f1 = try bmp_util.mallocPlane(f32, gpa, size.width, size.height);
    defer img_f1.free(gpa);
    const stride_f = img_f1.stride;

    bmp_util.copyByte2Float(img_src, img_f1, size);
    bmp_util.addFloatPlane(-128.0, img_f1, size);

    const src = try cuda.alloc(f32, img_f1.data.len);
    defer src.free();
    const dst = try cuda.alloc(f32, img_f1.data.len);
    defer dst.free();
    const device_stride = stride_f;

    try src.copyFrom(img_f1.data);

    var timer_cuda: Timer = .{ .io = io };

    const grid_full_warps: cuda.Dim3 = .{
        .x = size.width / common.ker2_block_width,
        .y = size.height / common.ker2_block_height,
        .z = 1,
    };
    const threads_full_warps: cuda.Dim3 = .{
        .x = 8,
        .y = common.ker2_block_width / 8,
        .z = common.ker2_block_height / 8,
    };
    const full_warps: cuda.LaunchConfig = .{ .grid = grid_full_warps, .block = threads_full_warps };

    const num_iterations = 100;

    var i: i32 = -1;
    while (i < num_iterations) : (i += 1) {
        if (i == 0) {
            try cuda.synchronize();
            timer_cuda.reset();
            timer_cuda.start();
        }

        try kernel2_dct.launch(full_warps, .{ dst, src, device_stride });
    }

    try cuda.synchronize();
    timer_cuda.stop();

    const avg_time = timer_cuda.total_ms / num_iterations;
    std.log.info("{d:.6} MPix/s //{d:.6} ms", .{
        (1e-6 * @as(f32, @floatFromInt(size.width * size.height))) / (1e-3 * avg_time),
        avg_time,
    });

    const small_blocks: cuda.LaunchConfig = .{
        .grid = .{ .x = size.width / block_size, .y = size.height / block_size },
        .block = .{ .x = block_size, .y = block_size },
    };

    try quantization_float.launch(small_blocks, .{ dst, device_stride });

    try kernel2_idct.launch(full_warps, .{ src, dst, device_stride });
    try cuda.synchronize();

    try src.copyTo(img_f1.data);

    bmp_util.addFloatPlane(128.0, img_f1, size);
    bmp_util.copyFloat2Byte(img_f1, img_dst, size);

    return avg_time;
}

/// Wrapper function for short CUDA version of DCT, quantization and IDCT implementations.
/// Returns execution time in milliseconds.
fn wrapperCudaShort(
    gpa: std.mem.Allocator,
    io: Io,
    module: cuda.Module,
    img_src: Plane(u8),
    img_dst: Plane(u8),
    size: Roi,
) !f32 {
    const kernel_short_dct = try module.function("cudaKernelShortDct");
    const kernel_short_idct = try module.function("cudaKernelShortIdct");
    const quantization_short = try module.function("cudaKernelQuantizationShort");

    const img_s1 = try bmp_util.mallocPlane(i16, gpa, size.width, size.height);
    defer img_s1.free(gpa);
    const stride_s = img_s1.stride;

    for (0..size.height) |i| {
        for (0..size.width) |j| {
            img_s1.data[i * stride_s + j] = @as(i16, img_src.data[i * img_src.stride + j]) - 128;
        }
    }

    const src_dst = try cuda.alloc(i16, img_s1.data.len);
    defer src_dst.free();
    const device_stride = stride_s;

    try src_dst.copyFrom(img_s1.data);

    var timer_lib_jpeg: Timer = .{ .io = io };

    const grid_short: cuda.Dim3 = .{
        .x = size.width / common.kers_block_width,
        .y = size.height / common.kers_block_height,
        .z = 1,
    };
    const threads_short: cuda.Dim3 = .{
        .x = 8,
        .y = common.kers_block_width / 8,
        .z = common.kers_block_height / 8,
    };
    const short_blocks: cuda.LaunchConfig = .{ .grid = grid_short, .block = threads_short };

    timer_lib_jpeg.start();
    try kernel_short_dct.launch(short_blocks, .{ src_dst, device_stride });
    try cuda.synchronize();
    timer_lib_jpeg.stop();

    const timer_lib_jpeg_span16b = timer_lib_jpeg.averageValue();

    const small_blocks: cuda.LaunchConfig = .{
        .grid = .{ .x = size.width / block_size, .y = size.height / block_size },
        .block = .{ .x = block_size, .y = block_size },
    };

    try quantization_short.launch(small_blocks, .{ src_dst, device_stride });

    try kernel_short_idct.launch(short_blocks, .{ src_dst, device_stride });
    try cuda.synchronize();

    try src_dst.copyTo(img_s1.data);

    for (0..size.height) |i| {
        for (0..size.width) |j| {
            const restored = @as(i32, img_s1.data[i * stride_s + j]) + 128;
            img_dst.data[i * img_dst.stride + j] = @intCast(bmp_util.clamp0255(restored));
        }
    }

    return timer_lib_jpeg_span16b;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("dct8x8_kernel.ptx"));
    defer module.unload();

    const sample_image_fname = "teapot512.bmp";
    const sample_image_fname_res_gold1 = "teapot512_gold1.bmp";
    const sample_image_fname_res_gold2 = "teapot512_gold2.bmp";
    const sample_image_fname_res_cuda1 = "teapot512_cuda1.bmp";
    const sample_image_fname_res_cuda2 = "teapot512_cuda2.bmp";
    const sample_image_fname_res_cuda_short = "teapot512_cuda_short.bmp";

    const sample_image = @embedFile(sample_image_fname);

    // preload image (acquire dimensions)
    std.log.info("CUDA sample DCT/IDCT implementation", .{});
    std.log.info("===================================", .{});
    std.log.info("Loading test image: {s}... ", .{sample_image_fname});
    const img_size = bmp_util.preLoadBmp(sample_image) catch {
        std.log.err("Error: Image file not found or invalid!", .{});
        std.process.exit(1);
    };
    const img_width = img_size.width;
    const img_height = img_size.height;

    // check image dimensions are multiples of block_size
    if (img_width % block_size != 0 or img_height % block_size != 0) {
        std.log.err("Error: Input image dimensions must be multiples of 8!", .{});
        std.process.exit(1);
    }

    std.log.info("[{d} x {d}]... ", .{ img_width, img_height });

    const img_src = try bmp_util.mallocPlane(u8, gpa, img_width, img_height);
    defer img_src.free(gpa);
    const img_dst_gold1 = try bmp_util.mallocPlane(u8, gpa, img_width, img_height);
    defer img_dst_gold1.free(gpa);
    const img_dst_gold2 = try bmp_util.mallocPlane(u8, gpa, img_width, img_height);
    defer img_dst_gold2.free(gpa);
    const img_dst_cuda1 = try bmp_util.mallocPlane(u8, gpa, img_width, img_height);
    defer img_dst_cuda1.free(gpa);
    const img_dst_cuda2 = try bmp_util.mallocPlane(u8, gpa, img_width, img_height);
    defer img_dst_cuda2.free(gpa);
    const img_dst_cuda_short = try bmp_util.mallocPlane(u8, gpa, img_width, img_height);
    defer img_dst_cuda_short.free(gpa);

    bmp_util.loadBmpAsGray(sample_image, img_size, img_src);

    std.log.info("Success", .{});
    std.log.info("Running Gold 1 (CPU) version... ", .{});
    _ = try wrapperGold1(gpa, io, img_src, img_dst_gold1, img_size);

    std.log.info("Success", .{});
    std.log.info("Running Gold 2 (CPU) version... ", .{});
    _ = try wrapperGold2(gpa, io, img_src, img_dst_gold2, img_size);

    std.log.info("Success", .{});
    std.log.info("Running CUDA 1 (GPU) version... ", .{});
    const time_cuda1 = try wrapperCuda1(gpa, io, module, img_src, img_dst_cuda1, img_size);

    std.log.info("Success", .{});
    std.log.info("Running CUDA 2 (GPU) version... ", .{});
    const time_cuda2 = try wrapperCuda2(gpa, io, module, img_src, img_dst_cuda2, img_size);

    std.log.info("Success", .{});
    std.log.info("Running CUDA short (GPU) version... ", .{});
    const time_cuda_short =
        try wrapperCudaShort(gpa, io, module, img_src, img_dst_cuda_short, img_size);

    std.log.info("Success", .{});
    std.log.info("Dumping result to {s}... ", .{sample_image_fname_res_gold1});
    try bmp_util.dumpBmpAsGray(gpa, io, sample_image_fname_res_gold1, img_dst_gold1, img_size);

    std.log.info("Success", .{});
    std.log.info("Dumping result to {s}... ", .{sample_image_fname_res_gold2});
    try bmp_util.dumpBmpAsGray(gpa, io, sample_image_fname_res_gold2, img_dst_gold2, img_size);

    std.log.info("Success", .{});
    std.log.info("Dumping result to {s}... ", .{sample_image_fname_res_cuda1});
    try bmp_util.dumpBmpAsGray(gpa, io, sample_image_fname_res_cuda1, img_dst_cuda1, img_size);

    std.log.info("Success", .{});
    std.log.info("Dumping result to {s}... ", .{sample_image_fname_res_cuda2});
    try bmp_util.dumpBmpAsGray(gpa, io, sample_image_fname_res_cuda2, img_dst_cuda2, img_size);

    std.log.info("Success", .{});
    std.log.info("Dumping result to {s}... ", .{sample_image_fname_res_cuda_short});
    try bmp_util.dumpBmpAsGray(
        gpa,
        io,
        sample_image_fname_res_cuda_short,
        img_dst_cuda_short,
        img_size,
    );

    std.log.info("Success", .{});

    std.log.info("Processing time (CUDA 1)    : {d:.6} ms ", .{time_cuda1});
    std.log.info("Processing time (CUDA 2)    : {d:.6} ms ", .{time_cuda2});
    std.log.info("Processing time (CUDA short): {d:.6} ms ", .{time_cuda_short});

    // calculate PSNR between each pair of images
    const psnr_src_dst_gold1 = bmp_util.calculatePsnr(img_src, img_dst_gold1, img_size);
    const psnr_src_dst_gold2 = bmp_util.calculatePsnr(img_src, img_dst_gold2, img_size);
    const psnr_src_dst_cuda1 = bmp_util.calculatePsnr(img_src, img_dst_cuda1, img_size);
    const psnr_src_dst_cuda2 = bmp_util.calculatePsnr(img_src, img_dst_cuda2, img_size);
    const psnr_src_dst_cuda_short = bmp_util.calculatePsnr(img_src, img_dst_cuda_short, img_size);
    const psnr_dst_gold1_dst_cuda1 = bmp_util.calculatePsnr(img_dst_gold1, img_dst_cuda1, img_size);
    const psnr_dst_gold2_dst_cuda2 = bmp_util.calculatePsnr(img_dst_gold2, img_dst_cuda2, img_size);
    const psnr_dst_gold2_dst_cuda16b =
        bmp_util.calculatePsnr(img_dst_gold2, img_dst_cuda_short, img_size);

    std.log.info("PSNR Original    <---> CPU(Gold 1)    : {d:.6}", .{psnr_src_dst_gold1});
    std.log.info("PSNR Original    <---> CPU(Gold 2)    : {d:.6}", .{psnr_src_dst_gold2});
    std.log.info("PSNR Original    <---> GPU(CUDA 1)    : {d:.6}", .{psnr_src_dst_cuda1});
    std.log.info("PSNR Original    <---> GPU(CUDA 2)    : {d:.6}", .{psnr_src_dst_cuda2});
    std.log.info("PSNR Original    <---> GPU(CUDA short): {d:.6}", .{psnr_src_dst_cuda_short});
    std.log.info("PSNR CPU(Gold 1) <---> GPU(CUDA 1)    : {d:.6}", .{psnr_dst_gold1_dst_cuda1});
    std.log.info("PSNR CPU(Gold 2) <---> GPU(CUDA 2)    : {d:.6}", .{psnr_dst_gold2_dst_cuda2});
    std.log.info("PSNR CPU(Gold 2) <---> GPU(CUDA short): {d:.6}", .{psnr_dst_gold2_dst_cuda16b});

    const test_result = psnr_dst_gold1_dst_cuda1 > psnr_threshold_equal and
        psnr_dst_gold2_dst_cuda2 > psnr_threshold_equal and
        psnr_dst_gold2_dst_cuda16b > psnr_threshold_equal;

    std.log.info("Test Summary...", .{});

    if (!test_result) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }

    std.log.info("Test passed", .{});
}
