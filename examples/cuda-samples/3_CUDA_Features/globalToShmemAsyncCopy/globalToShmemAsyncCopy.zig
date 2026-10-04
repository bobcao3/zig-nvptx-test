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

//! Matrix multiplication: C = A * B.
//!
//! This sample demonstrates implements matrix multiplication which makes use of
//! shared memory to ensure data reuse, the matrix multiplication is done using
//! tiling approach.
//! With compute capability 8.0 or higher the CUDA kernels involved uses
//! asynchronously copy data from global to shared memory; a.k.a., async-copy.
//! This sample has been written for clarity of exposition to illustrate various
//! CUDA programming principles, not with the goal of providing the most
//! performant generic kernel for matrix multiplication.

const std = @import("std");
const cuda = @import("cuda");

const Kernel = enum(u32) {
    async_copy_multi_stage_large_chunk = 0,
    async_copy_large_chunk = 1,
    async_copy_large_chunk_aw_barrier = 2,
    async_copy_multi_stage_shared_state = 3,
    async_copy_multi_stage = 4,
    async_copy_single_stage = 5,
    naive = 6,
    naive_large_chunk = 7,

    fn displayName(kernel: Kernel) []const u8 {
        return switch (kernel) {
            .async_copy_multi_stage_large_chunk => "AsyncCopyMultiStageLargeChunk",
            .async_copy_large_chunk => "AsyncCopyLargeChunk",
            .async_copy_large_chunk_aw_barrier => "AsyncCopyLargeChunkAWBarrier",
            .async_copy_multi_stage_shared_state => "AsyncCopyMultiStageSharedState",
            .async_copy_multi_stage => "AsyncCopyMultiStage",
            .async_copy_single_stage => "AsyncCopySingleStage",
            .naive => "Naive",
            .naive_large_chunk => "NaiveLargeChunk",
        };
    }

    fn functionName(kernel: Kernel) [:0]const u8 {
        return switch (kernel) {
            .async_copy_multi_stage_large_chunk => "matrixMulAsyncCopyMultiStageLargeChunk",
            .async_copy_large_chunk => "matrixMulAsyncCopyLargeChunk",
            .async_copy_large_chunk_aw_barrier => "matrixMulAsyncCopyLargeChunkAwBarrier",
            .async_copy_multi_stage_shared_state => "matrixMulAsyncCopyMultiStageSharedState",
            .async_copy_multi_stage => "matrixMulAsyncCopyMultiStage",
            .async_copy_single_stage => "matrixMulAsyncCopySingleStage",
            .naive => "matrixMulNaive",
            .naive_large_chunk => "matrixMulNaiveLargeChunk",
        };
    }
};

const Dims = struct { x: u32, y: u32 };

const block_size = 16;

fn constantInit(data: []f32, value: f32) void {
    @memset(data, value);
}

/// Runs matrix multiplication using CUDA and returns whether the result is correct.
fn matrixMultiply(gpa: std.mem.Allocator, dims_a: Dims, dims_b: Dims, kernel_number: Kernel) !bool {
    const size_a = dims_a.x * dims_a.y;
    const h_a = try gpa.alloc(f32, size_a);
    defer gpa.free(h_a);
    const size_b = dims_b.x * dims_b.y;
    const h_b = try gpa.alloc(f32, size_b);
    defer gpa.free(h_b);

    const val_b: f32 = 2.10;
    constantInit(h_a, 1.0);
    constantInit(h_b, val_b);

    const dims_c: Dims = .{ .x = dims_b.x, .y = dims_a.y };
    const h_c = try gpa.alloc(f32, dims_c.x * dims_c.y);
    defer gpa.free(h_c);

    const module = try cuda.Module.load(@embedFile("globalToShmemAsyncCopy_kernel.ptx"));
    defer module.unload();
    const matrix_mul = try module.function(kernel_number.functionName());

    const d_a = try cuda.alloc(f32, size_a);
    defer d_a.free();
    const d_b = try cuda.alloc(f32, size_b);
    defer d_b.free();
    const d_c = try cuda.alloc(f32, h_c.len);
    defer d_c.free();

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    const stream = try cuda.Stream.create();
    defer stream.destroy();

    try d_a.copyFromAsync(h_a, stream);
    try d_b.copyFromAsync(h_b, stream);
    try d_c.memsetAsync(0, stream);

    // The shared state kernel has a block of 16 x 18 work items: the first 16 rows are the
    // consumer group and the last 2 rows (one warp) are the producer group.
    const config: cuda.LaunchConfig = switch (kernel_number) {
        .async_copy_multi_stage_shared_state => .{
            .grid = .{ .x = dims_b.x / block_size, .y = dims_a.y / block_size },
            .block = .{ .x = block_size, .y = block_size + 2 },
            .stream = stream,
        },
        else => .{
            .grid = .{ .x = dims_b.x / block_size, .y = dims_a.y / block_size },
            .block = .{ .x = block_size, .y = block_size },
            .stream = stream,
        },
    };
    const args = .{ d_c, d_a, d_b, dims_a.x, dims_b.x };

    std.log.info("Running kernel = {d} - {s}", .{ @backingInt(kernel_number), kernel_number.displayName() });
    std.log.info("Computing result using CUDA Kernel...", .{});

    // A warmup launch.
    try matrix_mul.launch(config, args);
    std.log.info("done", .{});
    try stream.synchronize();

    const iterations = 100;

    try start.record(stream);
    for (0..iterations) |_| try matrix_mul.launch(config, args);
    try stop.record(stream);
    try stop.synchronize();

    const msec_total = try start.elapsed(stop);
    const msec_per_matrix_mul = msec_total / iterations;
    const flops_per_matrix_mul = 2.0 * @as(f64, @floatFromInt(dims_a.x)) *
        @as(f64, @floatFromInt(dims_a.y)) * @as(f64, @floatFromInt(dims_b.x));
    const giga_flops = (flops_per_matrix_mul * 1.0e-9) / (msec_per_matrix_mul / 1000.0);
    std.log.info("Performance= {d:.2} GFlop/s, Time= {d:.3} msec, Size= {d:.0} Ops, WorkgroupSize= {d} threads/block", .{
        giga_flops,
        msec_per_matrix_mul,
        flops_per_matrix_mul,
        block_size * block_size,
    });

    try d_c.copyToAsync(h_c, stream);
    try stream.synchronize();

    std.log.info("Checking computed result for correctness: ", .{});
    var correct = true;
    // Tests the relative error |<x, y>_cpu - <x, y>_gpu| / <|x|, |y|> < eps.
    const eps = 1.e-6;
    const reference = @as(f64, @floatFromInt(dims_a.x)) * val_b;
    for (h_c, 0..) |value, i| {
        const abs_err = @abs(value - reference);
        const dot_length: f64 = @floatFromInt(dims_a.x);
        const abs_val = @abs(value);
        const rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            std.log.err("Error! Matrix[{d:0>5}]={d:.8}, ref={d:.8} error term is > {e}", .{ i, value, reference, eps });
            correct = false;
        }
    }
    if (correct) std.log.info("Result = PASS", .{}) else std.log.err("Result = FAIL", .{});
    std.log.info("NOTE: The CUDA Samples are not meant for performance measurements. " ++
        "Results may vary when GPU Boost is enabled.", .{});
    return correct;
}

/// Returns the value of the command line argument `-<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) !?u32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') return try std.fmt.parseInt(u32, flag[name.len + 1 ..], 10);
    }
    return null;
}

fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, std.mem.trimStart(u8, arg, "-"), name)) return true;
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    std.log.info("[globalToShmemAsyncCopy] - Starting...", .{});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (hasFlag(args, "help") or hasFlag(args, "?")) {
        std.log.info(
            \\Usage -device=n (n >= 0 for deviceID)
            \\      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)
            \\      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)
            \\      -kernel=kernel_number (0 - AsyncCopyMultiStageLargeChunk; 1 - AsyncCopyLargeChunk)
            \\                            (2 - AsyncCopyLargeChunkAWBarrier; 3 - AsyncCopyMultiStageSharedState)
            \\                            (4 - AsyncCopyMultiStage; 5 - AsyncCopySingleStage; 6 - Naive without memcpy_async)
            \\                            (7 - NaiveLargeChunk without memcpy_async)
            \\  Note: Outer matrix dimensions of A & B matrices must be equal.
        , .{});
        return;
    }

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();

    const matrix_block = 32;
    var dims_a: Dims = .{ .x = 10 * 4 * matrix_block, .y = 10 * 4 * matrix_block };
    var dims_b: Dims = .{ .x = 10 * 4 * matrix_block, .y = 10 * 4 * matrix_block };
    if (try flagValue(args, "wA")) |value| dims_a.x = value;
    if (try flagValue(args, "hA")) |value| dims_a.y = value;
    if (try flagValue(args, "wB")) |value| dims_b.x = value;
    if (try flagValue(args, "hB")) |value| dims_b.y = value;

    if (dims_a.x != dims_b.y) {
        std.log.err("Error: outer matrix dimensions must be equal. ({d} != {d})", .{ dims_a.x, dims_b.y });
        std.process.exit(1);
    }

    var selected_kernel: Kernel = .async_copy_multi_stage_large_chunk;

    if (try flagValue(args, "kernel")) |kernel_number| {
        selected_kernel = std.enums.fromInt(Kernel, kernel_number) orelse {
            std.log.err("Error: kernel number should be between 0 to 6, you have entered {d}", .{kernel_number});
            std.process.exit(1);
        };
    }

    if ((try device.computeCapability()).major < 7) {
        std.log.warn("globalToShmemAsyncCopy requires SM 7.0 or higher.  Exiting...", .{});
        std.process.exit(2);
    }

    std.log.info("MatrixA({d},{d}), MatrixB({d},{d})", .{ dims_a.x, dims_a.y, dims_b.x, dims_b.y });

    if (!try matrixMultiply(init.gpa, dims_a, dims_b, selected_kernel)) std.process.exit(1);
}
