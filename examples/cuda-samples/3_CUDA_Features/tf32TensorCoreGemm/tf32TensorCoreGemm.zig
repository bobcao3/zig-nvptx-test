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

//! CUDA sample demonstrating a tf32 (E8M10) GEMM computation using the Warp Matrix Multiply
//! and Accumulate API introduced in CUDA 11.0.
//!
//! In this program, the computeTf32gemm kernel computes the result of a matrix multiplication
//! and addition: D = alpha * A * B + beta * C. The dimensions of both C and D matrices
//! are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x K_GLOBAL (row-major), the B matrix
//! is K_GLOBAL x N_GLOBAL (column-major).
//! In that kernel, each CTA computes one 128 x 128 tile of the resulting matrix
//! per iteration. When the tile is computed, the CTA stores it to the global memory
//! and begins a new iteration, selecting a new 128 x 128 tile to compute.
//! Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes eight
//! 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array.
//! Warps compute the 16 x 16 subtiles using wmma.mma operations by
//! moving through the K_GLOBAL dimension of the A and B matrices and accumulating
//! the intermediate result in the local thread state.
//!
//! There are a number of simple optimizations used in the algorithm:
//! - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//!   shared memory. After that is done, each warp loads the C matrix fragments from
//!   shared memory, thus avoiding a random global memory access.
//! - On each internal iteration, the CTA copies a portion of the A and B matrices from
//!   global memory to shared memory. After that, all warps in the CTA reuse the A and B
//!   data from shared memory, thus reducing the number of data copies from global memory.
//! - The portions of the A and B matrices are stored in shared memory with an additional
//!   padding (skew) to reduce the number of shared memory access bank conflicts.
//!   (See a detailed explanation near the skew_float constant.)
//! - When the CTA finishes computing the tiles of the resulting matrix, each warp stores
//!   its subtiles to shared memory. The CTA then copies the shared memory contents to
//!   global memory, again avoiding redundant random global memory accesses.
//! - Note that the CTA tile size is chosen to maximize the GPU register utilization,
//!   but carefully enough to avoid local memory use.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("tf32TensorCoreGemm_config.zig");

/// Set this to true to verify the correctness of the GPU-computed matrix.
const cpu_debug = false;

const Kernel = enum(u32) {
    /// Shared memory with `cp.async` copies.
    tf32mma_shmem_gemm_async_copy = 0,
    /// Shared memory with ordinary copies.
    tf32mma_shmem_gemm = 1,
    /// No shared memory.
    simple_tf32mma_gemm = 2,
};

const kernel_names = [_][]const u8{ "compute_tf32gemm_async_copy", "compute_tf32gemm", "simple_wmma_tf32gemm" };

fn initHostMatrices(random: std.Random, a: []f32, b: []f32, c: []f32) void {
    for (a) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 3));
    for (b) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 3));
    for (c) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 3));
}

fn matMultiplyOnHost(
    a: []const f32,
    b: []const f32,
    c: []f32,
    alpha: f32,
    beta: f32,
    num_a_columns: usize,
    num_b_rows: usize,
    num_c_rows: usize,
    num_c_columns: usize,
) void {
    for (0..num_c_rows) |i| {
        for (0..num_c_columns) |j| {
            var temp: f32 = 0.0;

            for (0..num_a_columns) |k| {
                temp += a[i * num_a_columns + k] * b[j * num_b_rows + k];
            }

            c[i * num_c_columns + j] = temp * alpha + beta * c[i * num_c_columns + j];
        }
    }
}

/// Returns the value of the command line argument `-<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) !?i32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') return try std.fmt.parseInt(i32, flag[name.len + 1 ..], 10);
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    std.log.info("Initializing...", .{});
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const gpa = init.gpa;

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();

    // Tensor cores require a GPU of Volta (SM8X) architecture or higher.
    const capability = try device.computeCapability();
    if (capability.major < 8) {
        std.log.warn("tf32TensorCoreGemm requires requires SM 8.0 or higher to use Tensor Cores.  Exiting...", .{});
        std.process.exit(2);
    }

    std.log.info("M: {d} ({d} x {d})", .{ config.m_global, config.m, config.m_tiles });
    std.log.info("N: {d} ({d} x {d})", .{ config.n_global, config.n, config.n_tiles });
    std.log.info("K: {d} ({d} x {d})", .{ config.k_global, config.k, config.k_tiles });

    const a_h = try gpa.alloc(f32, config.m_global * config.k_global);
    defer gpa.free(a_h);
    const b_h = try gpa.alloc(f32, config.k_global * config.n_global);
    defer gpa.free(b_h);
    const c_h = try gpa.alloc(f32, config.m_global * config.n_global);
    defer gpa.free(c_h);
    const result_hd = if (cpu_debug) try gpa.alloc(f32, config.m_global * config.n_global) else {};
    defer if (cpu_debug) gpa.free(result_hd);
    const result_host = if (cpu_debug) try gpa.alloc(f32, config.m_global * config.n_global) else {};
    defer if (cpu_debug) gpa.free(result_host);

    const module = try cuda.Module.load(@embedFile("tf32TensorCoreGemm_kernel.ptx"));
    defer module.unload();

    const a = try cuda.alloc(f32, config.m_global * config.k_global);
    defer a.free();
    const b = try cuda.alloc(f32, config.n_global * config.k_global);
    defer b.free();
    const c = try cuda.alloc(f32, config.m_global * config.n_global);
    defer c.free();
    const d = try cuda.alloc(f32, config.m_global * config.n_global);
    defer d.free();

    std.debug.assert(a.address % 128 == 0);
    std.debug.assert(b.address % 128 == 0);
    std.debug.assert(c.address % 128 == 0);
    std.debug.assert(d.address % 128 == 0);

    var prng: std.Random.DefaultPrng = .init(0);
    initHostMatrices(prng.random(), a_h, b_h, c_h);

    std.log.info("Preparing data for GPU...", .{});

    try a.copyFrom(a_h);
    try b.copyFrom(b_h);
    try c.copyFrom(c_h);
    try d.memset(0);

    // The kernels need shared memory for the C and D tiles of a work group, and for chunks of
    // the A and B matrices of a work group. The right amount to request is the larger of the two.
    const shmem_size = config.shmem_size;

    std.log.info("Required shared memory size: {d} Kb", .{shmem_size / 1024});

    const alpha: f32 = 1.1;
    const beta: f32 = 1.2;

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    try start.record(.default);

    // The kernel to run defaults to tf32mma_shmem_gemm_async_copy.
    var selected_kernel: Kernel = .tf32mma_shmem_gemm_async_copy;

    if (try flagValue(args, "kernel")) |kernel_number| {
        if (kernel_number >= 0 and kernel_number < 3) {
            selected_kernel = @fromBackingInt(@intCast(kernel_number));
        } else {
            std.log.err("Error: kernel number should be between 0 to 2, you have entered {d}", .{kernel_number});
            std.process.exit(1);
        }
    }

    const multiprocessor_count: u32 = @intCast(try device.attribute(.multiprocessor_count));
    const shared_memory_per_multiprocessor = try device.attribute(.max_shared_memory_per_multiprocessor);

    // If enough shared memory is available on the GPU, use the high performance kernel.
    if (shared_memory_per_multiprocessor >= shmem_size and selected_kernel != .simple_tf32mma_gemm) {
        std.log.info("Computing using high performance kernel = {d} - {s}", .{
            @backingInt(selected_kernel),
            kernel_names[@backingInt(selected_kernel)],
        });

        const function = try module.function(switch (selected_kernel) {
            .tf32mma_shmem_gemm_async_copy, .simple_tf32mma_gemm => "computeTf32gemmAsyncCopy",
            .tf32mma_shmem_gemm => "computeTf32gemm",
        });
        try function.setMaxDynamicSharedMemory(shmem_size);
        try function.launch(.{
            .grid = .{ .x = multiprocessor_count * 2 },
            .block = .{ .x = config.threads_per_block },
            .shared_memory_bytes = shmem_size,
        }, .{ a, b, c, d, alpha, beta });
        if (cpu_debug) try d.copyTo(result_hd);
    } else {
        // The block dimension x must be a multiple of the warp size. 128x4 means 16 warps, and a
        // block computes a 64x64 output tile.
        const block_x = 128;
        const block_y = 4;

        const grid_x = (config.m_global + (config.m * block_x / 32 - 1)) / (config.m * block_x / 32);
        const grid_y = (config.n_global + config.n * block_y - 1) / (config.n * block_y);

        std.log.info("Computing... using simple_wmma_gemm kernel", .{});
        const function = try module.function("simpleWmmaTf32gemm");
        try function.launch(.{
            .grid = .{ .x = grid_x, .y = grid_y },
            .block = .{ .x = block_x, .y = block_y },
        }, .{
            a,
            b,
            c,
            d,
            @as(i32, config.m_global),
            @as(i32, config.n_global),
            @as(i32, config.k_global),
            alpha,
            beta,
        });
        if (cpu_debug) try d.copyTo(result_hd);
    }

    try stop.record(.default);
    try stop.synchronize();

    if (cpu_debug) {
        std.log.info("Verifying correctness of the computations...", .{});

        @memcpy(result_host, c_h);

        matMultiplyOnHost(a_h, b_h, result_host, alpha, beta, config.k_global, config.k_global, config.m_global, config.n_global);

        for (result_hd, result_host, 0..) |gpu, host, i| {
            if (@abs(gpu - host) > 0.1) {
                std.log.err("mismatch i={d} result_hD={d:.6} result_host={d:.6}", .{ i, gpu, host });
            }
        }
    }

    const milliseconds = try start.elapsed(stop);

    std.log.info("Time: {d:.6} ms", .{milliseconds});
    const flops = @as(f64, config.m_global) * config.n_global * config.k_global * 2;
    std.log.info("TFLOPS: {d:.2}", .{(flops / (milliseconds / 1000.0)) / 1e12});
}
