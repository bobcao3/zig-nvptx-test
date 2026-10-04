// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

//! This example shows how to use the clock function to measure the performance of block of
//! threads of a kernel accurately. Blocks are executed in parallel and out of order. Since
//! there's no synchronization mechanism between blocks, we measure the clock once for each block.
//! The clock samples are written to device memory.

const std = @import("std");

const num_threads = 256;

var warp_minimums: [num_threads / std.nvptx.warp_size]f32 addrspace(.shared) = undefined;

/// Returns the minimum of the elements of all work items of the work group. Only work item 0
/// returns the minimum of the whole work group.
inline fn blockReduceMin(thread_data: [2]f32) f32 {
    const work_item: u32 = @workItemId(0);
    var minimum = @min(thread_data[0], thread_data[1]);
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        const other = std.nvptx.shflSync(.down, f32, minimum, offset, .fromWidth(.down, 32), .all);
        minimum = @min(minimum, other);
    }
    if (std.nvptx.laneid() == 0) warp_minimums[work_item / std.nvptx.warp_size] = minimum;
    std.nvptx.barrierSyncAligned(.work_group, null);

    if (work_item == 0) {
        for (warp_minimums[1..]) |warp_minimum| minimum = @min(minimum, warp_minimum);
    }
    return minimum;
}

/// Computes a standard parallel reduction and evaluates the time it takes to do that for each
/// block. The timing results are stored in device memory.
export fn timedReduction(
    input: [*]addrspace(.global) const f32,
    output: [*]addrspace(.global) f32,
    timer: [*]addrspace(.global) i64,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    const bid = @workGroupId(0);

    if (tid == 0) timer[bid] = @as(i32, @bitCast(std.nvptx.clock()));

    // Each work item loads 2 elements and reduces them to a local minimum.
    const thread_data = [2]f32{ input[tid], input[tid + @workGroupSize(0)] };

    const block_min = blockReduceMin(thread_data);

    if (tid == 0) output[bid] = block_min;

    std.nvptx.barrierSyncAligned(.work_group, null);

    if (tid == 0) timer[bid + std.nvptx.nctaid(0)] = @as(i32, @bitCast(std.nvptx.clock()));
}
