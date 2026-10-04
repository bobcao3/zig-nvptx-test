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

//! This sample implements a conjugate gradient solver on GPU using a multi-block cooperative
//! launch, and also uses unified memory.

const std = @import("std");

const GridBarrier = std.nvptx.GridBarrier;

const threads_per_block = 512;

extern var tmp: [0]f64 align(16) addrspace(.shared);
var grid_counter: u32 addrspace(.global) = 0;

inline fn shflDown(value: f64, offset: u32) f64 {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.down, std.nvptx.warp_size);
    const words: [2]u32 = @bitCast(value);
    return @bitCast([2]u32{
        std.nvptx.shflSync(.down, u32, words[0], offset, clamp, .all),
        std.nvptx.shflSync(.down, u32, words[1], offset, clamp, .all),
    });
}

inline fn tileReduceSum(value: f64) f64 {
    var sum = value;
    var offset: u32 = std.nvptx.warp_size / 2;
    while (offset > 0) : (offset /= 2) sum += shflDown(sum, offset);
    return sum;
}

inline fn gridThreadRank() u32 {
    return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
}

inline fn gridSize() u32 {
    return std.nvptx.nctaid(0) * @workGroupSize(0);
}

inline fn gpuSpMV(
    i_rows: [*]addrspace(.global) const i32,
    j_columns: [*]addrspace(.global) const i32,
    val: [*]addrspace(.global) const f32,
    num_rows: i32,
    alpha: f32,
    input_vec_x: [*]addrspace(.global) const f32,
    output_vec_y: [*]addrspace(.global) f32,
) void {
    var i = gridThreadRank();
    while (i < num_rows) : (i += gridSize()) {
        const row_elem: u32 = @intCast(i_rows[i]);
        const next_row_elem: u32 = @intCast(i_rows[i + 1]);
        const num_elems_this_row = next_row_elem - row_elem;

        var output: f32 = 0.0;
        for (0..num_elems_this_row) |j| {
            output += alpha * val[row_elem + j] * input_vec_x[@intCast(j_columns[row_elem + j])];
        }

        output_vec_y[i] = output;
    }
}

inline fn gpuSaxpy(
    x: [*]addrspace(.global) const f32,
    y: [*]addrspace(.global) f32,
    a: f32,
    size: i32,
) void {
    var i = gridThreadRank();
    while (i < size) : (i += gridSize()) {
        y[i] = a * x[i] + y[i];
    }
}

inline fn gpuDotProduct(
    vec_a: [*]addrspace(.global) const f32,
    vec_b: [*]addrspace(.global) const f32,
    result: *addrspace(.global) f64,
    size: i32,
) void {
    const shared: [*]addrspace(.shared) f64 = @ptrCast(&tmp);
    const thread_rank = @workItemId(0) % std.nvptx.warp_size;
    const meta_group_rank = @workItemId(0) / std.nvptx.warp_size;
    const meta_group_size = (@workGroupSize(0) + std.nvptx.warp_size - 1) / std.nvptx.warp_size;

    var temp_sum: f64 = 0.0;
    var i = gridThreadRank();
    while (i < size) : (i += gridSize()) {
        temp_sum += vec_a[i] * vec_b[i];
    }

    temp_sum = tileReduceSum(temp_sum);

    if (thread_rank == 0) shared[meta_group_rank] = temp_sum;

    std.nvptx.barrierSyncAligned(.work_group, null);

    if (meta_group_rank == 0) {
        temp_sum = if (thread_rank < meta_group_size) shared[thread_rank] else 0.0;
        temp_sum = tileReduceSum(temp_sum);

        if (thread_rank == 0) _ = @atomicRmw(f64, result, .Add, temp_sum, .monotonic);
    }
}

inline fn gpuCopyVector(
    src_a: [*]addrspace(.global) const f32,
    dest_b: [*]addrspace(.global) f32,
    size: i32,
) void {
    var i = gridThreadRank();
    while (i < size) : (i += gridSize()) {
        dest_b[i] = src_a[i];
    }
}

inline fn gpuScaleVectorAndSaxpy(
    x: [*]addrspace(.global) const f32,
    y: [*]addrspace(.global) f32,
    a: f32,
    scale: f32,
    size: i32,
) void {
    var i = gridThreadRank();
    while (i < size) : (i += gridSize()) {
        y[i] = a * x[i] + scale * y[i];
    }
}

export fn gpuConjugateGradient(
    i_rows: [*]addrspace(.global) const i32,
    j_columns: [*]addrspace(.global) const i32,
    val: [*]addrspace(.global) const f32,
    x: [*]addrspace(.global) f32,
    ax: [*]addrspace(.global) f32,
    p: [*]addrspace(.global) f32,
    r: [*]addrspace(.global) f32,
    dot_result: *addrspace(.global) f64,
    nnz: i32,
    n: i32,
    tol: f32,
) callconv(.{ .nvptx_kernel = .{ .max_work_group_size = .{ .x = threads_per_block } } }) void {
    _ = nnz;
    const grid_barrier: GridBarrier = .{ .counter = &grid_counter };
    const max_iter = 10000;

    const alpha: f32 = 1.0;
    const alpham1: f32 = -1.0;
    var r0: f32 = 0.0;
    var r1: f32 = undefined;
    var b: f32 = undefined;
    var a: f32 = undefined;
    var na: f32 = undefined;

    gpuSpMV(i_rows, j_columns, val, n, alpha, x, ax);

    grid_barrier.sync();

    gpuSaxpy(ax, r, alpham1, n);

    grid_barrier.sync();

    gpuDotProduct(r, r, dot_result, n);

    grid_barrier.sync();

    r1 = @floatCast(dot_result.*);

    var k: u32 = 1;
    while (r1 > tol * tol and k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            gpuScaleVectorAndSaxpy(r, p, alpha, b, n);
        } else {
            gpuCopyVector(r, p, n);
        }

        grid_barrier.sync();

        gpuSpMV(i_rows, j_columns, val, n, alpha, p, ax);

        if (@workItemId(0) == 0 and @workGroupId(0) == 0) dot_result.* = 0.0;

        grid_barrier.sync();

        gpuDotProduct(p, ax, dot_result, n);

        grid_barrier.sync();

        a = @floatCast(@as(f64, r1) / dot_result.*);

        gpuSaxpy(p, x, a, n);
        na = -a;
        gpuSaxpy(ax, r, na, n);

        r0 = r1;

        grid_barrier.sync();
        if (@workItemId(0) == 0 and @workGroupId(0) == 0) dot_result.* = 0.0;

        grid_barrier.sync();

        gpuDotProduct(r, r, dot_result, n);

        grid_barrier.sync();

        r1 = @floatCast(dot_result.*);
        k += 1;
    }
}
