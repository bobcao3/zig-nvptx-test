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

//! A simple demonstration of arrive-wait barriers.
//!
//! This sample uses an arrive-wait barrier in shared memory to reduce the partial dot products of
//! the work items of a work group, and a grid-wide barrier to combine the results of all work
//! groups. The kernel then normalizes two vectors by the square root of their dot product.

const std = @import("std");

const GridBarrier = std.nvptx.GridBarrier;
const Mbarrier = std.nvptx.Mbarrier;

extern var tmp: [0]f64 align(16) addrspace(.shared);
var barrier: Mbarrier addrspace(.shared) = undefined;
var grid_counter: u32 addrspace(.global) = 0;

inline fn shflDown(value: f64, offset: u32) f64 {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.down, std.nvptx.warp_size);
    const words: [2]u32 = @bitCast(value);
    return @bitCast([2]u32{
        std.nvptx.shflSync(.down, u32, words[0], offset, clamp, .all),
        std.nvptx.shflSync(.down, u32, words[1], offset, clamp, .all),
    });
}

inline fn reduceBlockData(
    comptime write_square_root: bool,
    thread_sum_in: f64,
    result: *addrspace(.global) f64,
) void {
    const tmp_ptr: [*]addrspace(.shared) f64 = @ptrCast(&tmp);
    const thread_rank = @workItemId(0) % std.nvptx.warp_size;
    const meta_group_rank = @workItemId(0) / std.nvptx.warp_size;
    const meta_group_size = (@workGroupSize(0) + std.nvptx.warp_size - 1) / std.nvptx.warp_size;

    var thread_sum = thread_sum_in;
    var offset: u32 = std.nvptx.warp_size / 2;
    while (offset > 0) : (offset /= 2) thread_sum += shflDown(thread_sum, offset);
    if (thread_rank == 0) tmp_ptr[meta_group_rank] = thread_sum;

    const token = barrier.arrive();
    while (!barrier.testWait(token)) {}

    if (meta_group_rank == 0) {
        var beta: f64 = if (thread_rank < meta_group_size) tmp_ptr[thread_rank] else 0.0;

        offset = std.nvptx.warp_size / 2;
        while (offset > 0) : (offset /= 2) beta += shflDown(beta, offset);

        if (thread_rank == 0) {
            result.* = if (write_square_root) @sqrt(beta) else beta;
        }
    }
}

export fn normVecByDotProductAwBarrier(
    vec_a: [*]addrspace(.global) f32,
    vec_b: [*]addrspace(.global) f32,
    partial_results: [*]addrspace(.global) f64,
    size: i32,
) callconv(.kernel) void {
    const grid_barrier: GridBarrier = .{ .counter = &grid_counter };
    const grid_size = std.nvptx.nctaid(0) * @workGroupSize(0);
    const grid_thread_rank = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const length: u32 = @intCast(size);

    if (@workItemId(0) == 0) barrier.init(@workGroupSize(0));

    std.nvptx.barrierSyncAligned(.work_group, null);

    var thread_sum: f64 = 0.0;
    var i = grid_thread_rank;
    while (i < length) : (i += grid_size) {
        thread_sum += vec_a[i] * vec_b[i];
    }

    reduceBlockData(false, thread_sum, &partial_results[@workGroupId(0)]);

    grid_barrier.sync();

    if (@workGroupId(0) == 0) {
        thread_sum = 0.0;
        i = @workItemId(0);
        while (i < std.nvptx.nctaid(0)) : (i += @workGroupSize(0)) {
            thread_sum += partial_results[i];
        }
        reduceBlockData(true, thread_sum, &partial_results[0]);
    }

    grid_barrier.sync();

    const final_value = partial_results[0];

    i = grid_thread_rank;
    while (i < length) : (i += grid_size) {
        vec_a[i] = @floatCast(@as(f64, vec_a[i]) / final_value);
        vec_b[i] = @floatCast(@as(f64, vec_b[i]) / final_value);
    }
}
