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

//! CUDA sample demonstrating an integer GEMM computation using the Warp Matrix Multiply and
//! Accumulate API.
//!
//! In this program, the `computeGemmImma` kernel computes the result of a matrix multiplication
//! and addition: D = alpha * A * B + beta * C. The dimensions of both C and D matrices are
//! `m_global` x `n_global`. The A matrix is `m_global` x `k_global` (row-major), the B matrix is
//! `k_global` x `n_global` (column-major). In that kernel, each work group computes one 128 x 128
//! tile of the resulting matrix per iteration. When the tile is computed, the work group stores
//! it to global memory and begins a new iteration, selecting a new 128 x 128 tile to compute.

const std = @import("std");
const cuda = @import("cuda");
const constants = @import("immaTensorCoreGemm_constants.zig");

const m = constants.m;
const n = constants.n;
const k = constants.k;
const m_global = constants.m_global;
const n_global = constants.n_global;
const k_global = constants.k_global;

/// Set this to true to verify the correctness of the matrix that the GPU computes.
const cpu_debug = false;

fn initHostMatrices(random: std.Random, a: []u8, b: []u8, c: []i32) void {
    for (a) |*element| element.* = @intCast(random.uintLessThan(u32, 3));
    for (b) |*element| element.* = @intCast(random.uintLessThan(u32, 3));
    for (c) |*element| element.* = @intCast(random.uintLessThan(u32, 3));
}

fn matMultiplyOnHost(
    a: []const u8,
    b: []const u8,
    c: []i32,
    alpha: i32,
    beta: i32,
    num_a_columns: usize,
    num_b_rows: usize,
    num_c_columns: usize,
) void {
    for (0..m_global) |i| {
        for (0..num_c_columns) |j| {
            var temp: i32 = 0;

            for (0..num_a_columns) |l| {
                temp += @as(i32, a[i * num_a_columns + l]) * @as(i32, b[j * num_b_rows + l]);
            }

            c[i * num_c_columns + j] = temp * alpha + beta * c[i * num_c_columns + j];
        }
    }
}

pub fn main(init: std.process.Init) !void {
    std.log.info("Initializing...", .{});

    const device = try cuda.init(0);
    defer device.deinit();

    const compute_capability = try device.computeCapability();
    const waived = compute_capability.major < 7 or
        (compute_capability.major <= 7 and compute_capability.minor < 2);
    if (waived) {
        std.log.warn(
            "immaTensorCoreGemm requires SM 7.2 or higher to use Tensor Cores.  Exiting...",
            .{},
        );
        return;
    }

    std.log.info("M: {d} ({d} x {d})", .{ m_global, m, constants.m_tiles });
    std.log.info("N: {d} ({d} x {d})", .{ n_global, n, constants.n_tiles });
    std.log.info("K: {d} ({d} x {d})", .{ k_global, k, constants.k_tiles });

    const a_h = try init.gpa.alloc(u8, m_global * k_global);
    defer init.gpa.free(a_h);
    const b_h = try init.gpa.alloc(u8, k_global * n_global);
    defer init.gpa.free(b_h);
    const c_h = try init.gpa.alloc(i32, m_global * n_global);
    defer init.gpa.free(c_h);
    const result_hd = if (cpu_debug) try init.gpa.alloc(i32, m_global * n_global);
    defer if (cpu_debug) init.gpa.free(result_hd);
    const result_host = if (cpu_debug) try init.gpa.alloc(i32, m_global * n_global);
    defer if (cpu_debug) init.gpa.free(result_host);

    const module = try cuda.Module.load(@embedFile("immaTensorCoreGemm_kernel.ptx"));
    defer module.unload();
    const compute_gemm_imma = try module.function("computeGemmImma");
    const simple_wmma_gemm_imma = try module.function("simpleWmmaGemmImma");

    const a = try cuda.alloc(u8, m_global * k_global);
    defer a.free();
    const b = try cuda.alloc(u8, n_global * k_global);
    defer b.free();
    const c = try cuda.alloc(i32, m_global * n_global);
    defer c.free();
    const d = try cuda.alloc(i32, m_global * n_global);
    defer d.free();

    for ([_]u64{ a.address, b.address, c.address, d.address }) |address| {
        std.debug.assert(address % 128 == 0);
    }

    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.real.now(init.io).toSeconds()));
    initHostMatrices(prng.random(), a_h, b_h, c_h);

    try a.copyFrom(a_h);
    try b.copyFrom(b_h);
    try c.copyFrom(c_h);
    try d.memset(0);

    std.log.info("Preparing data for GPU...", .{});

    const shared_memory_bytes = constants.shared_memory_bytes;
    std.log.info("Required shared memory size: {d} Kb", .{shared_memory_bytes / 1024});

    const alpha: i32 = 1;
    const beta: i32 = 1;

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    try start.record(.default);

    const shared_memory_per_multiprocessor =
        try device.attribute(.max_shared_memory_per_multiprocessor);
    if (shared_memory_per_multiprocessor >= shared_memory_bytes) {
        std.log.info("Computing... using high performance kernel compute_gemm_imma ", .{});

        try compute_gemm_imma.setMaxDynamicSharedMemory(shared_memory_bytes);
        try compute_gemm_imma.launch(.{
            .grid = .{ .x = @intCast(try device.attribute(.multiprocessor_count)) },
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

        std.log.info("Computing... using simple_wmma_gemm_imma kernel", .{});
        try simple_wmma_gemm_imma.launch(.{
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

        matMultiplyOnHost(a_h, b_h, result_host, alpha, beta, k_global, k_global, n_global);

        for (result_hd, result_host, 0..) |computed, expected, i| {
            if (computed != expected) {
                std.log.err(
                    "mismatch i={d} result_hD={d} result_host={d}",
                    .{ i, computed, expected },
                );
                correct = false;
            }
        }
    }

    const milliseconds = try start.elapsed(stop);

    std.log.info("Time: {d:.6} ms", .{milliseconds});
    const ops = @as(f64, m_global) * n_global * k_global * 2;
    std.log.info("TOPS: {d:.2}", .{ops / (milliseconds / 1000.0) / 1e12});

    if (!correct) std.process.exit(1);
}
