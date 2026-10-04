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

//! Walsh transforms belong to a class of generalized Fourier transformations. They have
//! applications in various fields of electrical engineering and numeric theory. In this sample we
//! demonstrate efficient implementation of naturally-ordered Walsh transform (also known as
//! Walsh-Hadamard or Hadamard transform) in CUDA and its particular application to dyadic
//! convolution computation. Refer to excellent Jorg Arndt's "Algorithms for Programmers"
//! textbook http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
//!
//! Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)

const std = @import("std");
const cuda = @import("cuda");
const gold = @import("fastWalshTransform_gold.zig");

const elementary_log2_size = 11;

const log2_kernel = 7;
const log2_data = 23;

const data_n = 1 << log2_data;
const kernel_n = 1 << log2_kernel;

const nops = 3.0 * @as(f64, data_n) * @as(f64, log2_data) / 2.0;

const Fwt = struct {
    fwt_batch1_kernel: cuda.Function,
    fwt_batch2_kernel: cuda.Function,
    modulate_kernel: cuda.Function,

    /// Batched Fast Walsh Transform.
    fn fwtBatchGpu(fwt: Fwt, d_data: cuda.Buffer(f32), batches: u32, log2n_in: u32) !void {
        const thread_n = 256;

        var m = batches;
        var log2n = log2n_in;
        var n: u32 = @as(u32, 1) << @intCast(log2n);
        const grid: cuda.Dim3 = .{ .x = (@as(u32, 1) << @intCast(log2n)) / (4 * thread_n), .y = m };

        while (log2n > elementary_log2_size) : ({
            log2n -= 2;
            n >>= 2;
            m <<= 2;
        }) {
            try fwt.fwt_batch2_kernel.launch(
                .{ .grid = grid, .block = .{ .x = thread_n } },
                .{ d_data, d_data, n / 4 },
            );
        }

        try fwt.fwt_batch1_kernel.launch(.{
            .grid = .{ .x = m },
            .block = .{ .x = n / 4 },
            .shared_memory_bytes = n * @sizeOf(f32),
        }, .{ d_data, d_data, log2n });
    }

    fn modulateGpu(fwt: Fwt, d_a: cuda.Buffer(f32), d_b: cuda.Buffer(f32), n: u32) !void {
        try fwt.modulate_kernel.launch(
            .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } },
            .{ d_a, d_b, n },
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();

    std.log.info("Initializing data...", .{});
    std.log.info("...allocating CPU memory", .{});
    const h_kernel = try init.gpa.alloc(f32, kernel_n);
    defer init.gpa.free(h_kernel);
    const h_data = try init.gpa.alloc(f32, data_n);
    defer init.gpa.free(h_data);
    const h_result_cpu = try init.gpa.alloc(f32, data_n);
    defer init.gpa.free(h_result_cpu);
    const h_result_gpu = try init.gpa.alloc(f32, data_n);
    defer init.gpa.free(h_result_gpu);
    std.log.info("...allocating GPU memory", .{});
    const d_kernel = try cuda.alloc(f32, data_n);
    defer d_kernel.free();
    const d_data = try cuda.alloc(f32, data_n);
    defer d_data.free();

    std.log.info("...generating data", .{});
    std.log.info("Data length: {d}; kernel length: {d}", .{ data_n, kernel_n });
    var prng: std.Random.DefaultPrng = .init(2007);
    const random = prng.random();

    for (h_kernel) |*element| element.* = random.float(f32);
    for (h_data) |*element| element.* = random.float(f32);

    try d_kernel.memset(0);
    try d_kernel.copyFrom(h_kernel);
    try d_data.copyFrom(h_data);

    const module = try cuda.Module.load(@embedFile("fastWalshTransform_kernel.ptx"));
    defer module.unload();
    const fwt: Fwt = .{
        .fwt_batch1_kernel = try module.function("fwtBatch1Kernel"),
        .fwt_batch2_kernel = try module.function("fwtBatch2Kernel"),
        .modulate_kernel = try module.function("modulateKernel"),
    };

    std.log.info("Running GPU dyadic convolution using Fast Walsh Transform...", .{});
    try cuda.synchronize();
    const start = std.Io.Clock.awake.now(init.io);
    try fwt.fwtBatchGpu(d_data, 1, log2_data);
    try fwt.fwtBatchGpu(d_kernel, 1, log2_data);
    try fwt.modulateGpu(d_data, d_kernel, data_n);
    try fwt.fwtBatchGpu(d_data, 1, log2_data);
    try cuda.synchronize();
    const gpu_time = elapsedSeconds(init.io, start) * 1000.0;
    std.log.info("GPU time: {d:.6} ms; GOP/s: {d:.6}", .{ gpu_time, nops / (gpu_time * 0.001 * 1e9) });

    std.log.info("Reading back GPU results...", .{});
    try d_data.copyTo(h_result_gpu);

    std.log.info("Running straightforward CPU dyadic convolution...", .{});
    gold.dyadicConvolutionCpu(h_result_cpu, h_data, h_kernel, log2_data, log2_kernel);

    std.log.info("Comparing the results...", .{});
    var sum_delta2: f64 = 0;
    var sum_ref2: f64 = 0;

    for (h_result_cpu, h_result_gpu) |cpu, gpu| {
        const delta: f64 = cpu - gpu;
        const ref: f64 = cpu;
        sum_delta2 += delta * delta;
        sum_ref2 += ref * ref;
    }

    const l2_norm = @sqrt(sum_delta2 / sum_ref2);

    std.log.info("Shutting down...", .{});

    std.log.info("L2 norm: {E}", .{l2_norm});
    const passed = l2_norm < 1e-6;
    if (passed) std.log.info("Test passed", .{}) else std.log.err("Test failed!", .{});
    if (!passed) std.process.exit(1);
}

fn elapsedSeconds(io: std.Io, start: std.Io.Timestamp) f64 {
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io));
    return @as(f64, @floatFromInt(elapsed.toNanoseconds())) * 1.0e-9;
}
