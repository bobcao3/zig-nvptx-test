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

//! CUDA sample demonstrating a Double precision GEMM computation using the Warp Matrix Multiply
//! and Accumulate API introduced in CUDA 11.0.
//!
//! In this program, the `computeDgemm` kernel computes the result of a matrix multiplication and
//! addition: D = alpha * A * B + beta * C. The dimensions of both C and D matrices are `m_global`
//! x `n_global`. The A matrix is `m_global` x `k_global` (row-major), the B matrix is `k_global` x
//! `n_global` (column-major). In that kernel, each work group computes one 64 x 64 tile of the
//! resulting matrix per iteration. When the tile is computed, the work group stores it to global
//! memory and begins a new iteration, selecting a new 64 x 64 tile to compute.

const std = @import("std");
const cuda = @import("cuda");
const constants = @import("dmmaTensorCoreGemm_constants.zig");

const m = constants.m;
const n = constants.n;
const k = constants.k;
const m_global = constants.m_global;
const n_global = constants.n_global;
const k_global = constants.k_global;

/// Set this to true to verify the correctness of the matrix that the GPU computes.
const cpu_debug = false;

const Kernel = enum {
    /// A kernel with asynchronous copies through a pipeline.
    dmma_shmem_gemm_async_copy,
    /// A kernel with asynchronous copies through cooperative groups.
    dmma_shmem_gemm_cg_async_copy,
    /// A kernel with ordinary copies.
    dmma_shmem_gemm,
    /// A kernel without shared memory.
    simple_dmma_gemm,
};

const kernel_names = [_][:0]const u8{
    "compute_dgemm_async_copy",
    "compute_dgemm_cg_async_copy",
    "compute_dgemm",
    "simple_wmma_gemm",
};

fn initHostMatrices(random: std.Random, a: []f64, b: []f64, c: []f64) void {
    for (a) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 3));
    for (b) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 3));
    for (c) |*element| element.* = @floatFromInt(random.uintLessThan(u32, 3));
}

fn matMultiplyOnHost(
    a: []const f64,
    b: []const f64,
    c: []f64,
    alpha: f32,
    beta: f32,
    num_a_columns: usize,
    num_b_rows: usize,
    num_c_columns: usize,
) void {
    for (0..m_global) |i| {
        for (0..num_c_columns) |j| {
            var temp: f64 = 0.0;

            for (0..num_a_columns) |l| {
                // The B matrix is column-major. The A matrix is row-major.
                temp += a[i * num_a_columns + l] * b[j * num_b_rows + l];
            }

            c[i * num_c_columns + j] = temp * alpha + beta * c[i * num_c_columns + j];
        }
    }
}

/// Returns the value of the command line argument `-<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) !?u32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') {
            return try std.fmt.parseInt(u32, flag[name.len + 1 ..], 10);
        }
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    std.log.info("Initializing...", .{});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const device = try cuda.init(0);
    defer device.deinit();

    const compute_capability = try device.computeCapability();
    if (compute_capability.major < 8) {
        std.log.warn("dmmaTensorCoreGemm requires SM 8.0 or higher.  Exiting...", .{});
        return;
    }

    std.log.info("M: {d} ({d} x {d})", .{ m_global, m, constants.m_tiles });
    std.log.info("N: {d} ({d} x {d})", .{ n_global, n, constants.n_tiles });
    std.log.info("K: {d} ({d} x {d})", .{ k_global, k, constants.k_tiles });

    const a_h = try init.gpa.alloc(f64, m_global * k_global);
    defer init.gpa.free(a_h);
    const b_h = try init.gpa.alloc(f64, k_global * n_global);
    defer init.gpa.free(b_h);
    const c_h = try init.gpa.alloc(f64, m_global * n_global);
    defer init.gpa.free(c_h);
    const result_hd = if (cpu_debug) try init.gpa.alloc(f64, m_global * n_global);
    defer if (cpu_debug) init.gpa.free(result_hd);
    const result_host = if (cpu_debug) try init.gpa.alloc(f64, m_global * n_global);
    defer if (cpu_debug) init.gpa.free(result_host);

    const module = try cuda.Module.load(@embedFile("dmmaTensorCoreGemm_kernel.ptx"));
    defer module.unload();

    const a = try cuda.alloc(f64, m_global * k_global);
    defer a.free();
    const b = try cuda.alloc(f64, n_global * k_global);
    defer b.free();
    const c = try cuda.alloc(f64, m_global * n_global);
    defer c.free();
    const d = try cuda.alloc(f64, m_global * n_global);
    defer d.free();

    for ([_]u64{ a.address, b.address, c.address, d.address }) |address| {
        std.debug.assert(address % 128 == 0);
    }

    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.real.now(init.io).toSeconds()));
    initHostMatrices(prng.random(), a_h, b_h, c_h);

    std.log.info("Preparing data for GPU...", .{});

    try a.copyFrom(a_h);
    try b.copyFrom(b_h);
    try c.copyFrom(c_h);
    try d.memset(0);

    const shared_memory_bytes = constants.shared_memory_bytes;
    std.log.info("Required shared memory size: {d} Kb", .{shared_memory_bytes / 1024});

    const alpha: f64 = @as(f32, 1.1);
    const beta: f64 = @as(f32, 1.2);

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    try start.record(.default);

    var selected_kernel: Kernel = .dmma_shmem_gemm_async_copy;

    if (try flagValue(args, "kernel")) |kernel_number| {
        if (kernel_number < kernel_names.len) {
            selected_kernel = @fromBackingInt(@intCast(kernel_number));
        } else {
            std.log.err(
                "Error: kernel number should be between 0 to 3, you have entered {d}",
                .{kernel_number},
            );
            std.process.exit(1);
        }
    }

    const shared_memory_per_multiprocessor =
        try device.attribute(.max_shared_memory_per_multiprocessor);
    const multiprocessor_count: u32 = @intCast(try device.attribute(.multiprocessor_count));
    const shared_memory_suffices = shared_memory_per_multiprocessor >= shared_memory_bytes;
    if (shared_memory_suffices and selected_kernel != .simple_dmma_gemm) {
        std.log.info("Computing using high performance kernel = {d} - {s}", .{
            @backingInt(selected_kernel),
            kernel_names[@backingInt(selected_kernel)],
        });

        const kernel = try module.function(switch (selected_kernel) {
            .dmma_shmem_gemm_async_copy => "computeDgemmAsyncCopy",
            .dmma_shmem_gemm_cg_async_copy => "computeDgemmCgAsyncCopy",
            .dmma_shmem_gemm => "computeDgemm",
            .simple_dmma_gemm => unreachable,
        });
        const blocks_per_multiprocessor: u32 = switch (selected_kernel) {
            .dmma_shmem_gemm => 2,
            else => 3,
        };
        try kernel.setMaxDynamicSharedMemory(shared_memory_bytes);
        try kernel.launch(.{
            .grid = .{ .x = multiprocessor_count * blocks_per_multiprocessor },
            .block = .{ .x = constants.threads_per_block },
            .shared_memory_bytes = shared_memory_bytes,
        }, .{ a, b, c, d, alpha, beta });

        if (cpu_debug) try d.copyTo(result_hd);
    } else {
        // The x dimension of a work group must be a multiple of the warp size. A work group of
        // 128 x 4 has 16 warps and computes a 64 x 64 output tile.
        const block_x = 128;
        const block_y = 4;

        const grid_x = (m_global + (m * block_x / 32 - 1)) / (m * block_x / 32);
        const grid_y = (n_global + n * block_y - 1) / (n * block_y);

        std.log.info("Computing... using simple_wmma_gemm kernel", .{});
        const simple_wmma_gemm = try module.function("simpleWmmaGemm");
        try simple_wmma_gemm.launch(.{
            .grid = .{ .x = grid_x, .y = grid_y },
            .block = .{ .x = block_x, .y = block_y },
        }, .{
            a,
            b,
            c,
            d,
            @as(u32, m_global),
            @as(u32, n_global),
            @as(u32, k_global),
            alpha,
            beta,
        });
        if (cpu_debug) try d.copyTo(result_hd);
    }

    try stop.record(.default);
    try stop.synchronize();

    var correct = true;
    if (cpu_debug) {
        std.log.info("Verifying correctness of the computations...", .{});

        @memcpy(result_host, c_h);

        matMultiplyOnHost(
            a_h,
            b_h,
            result_host,
            @floatCast(alpha),
            @floatCast(beta),
            k_global,
            k_global,
            n_global,
        );

        var number_of_matches: usize = 0;
        for (result_hd, result_host, 0..) |computed, expected, i| {
            if (@abs(computed - expected) > 0.1) {
                std.log.err(
                    "mismatch i={d} result_hD={d:.6} result_host={d:.6}",
                    .{ i, computed, expected },
                );
                correct = false;
                break;
            }
            number_of_matches += 1;
        }
        std.log.info(
            "number_of_matches = {d} out of = {d} ",
            .{ number_of_matches, n_global * m_global },
        );
    }

    const milliseconds = try start.elapsed(stop);

    std.log.info("Time: {d:.6} ms", .{milliseconds});
    const flops = @as(f64, m_global) * n_global * k_global * 2;
    std.log.info("FP64 TFLOPS: {d:.2}", .{flops / (milliseconds / 1000.0) / 1e12});

    if (!correct) std.process.exit(1);
}
