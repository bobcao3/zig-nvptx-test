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

//! Parallel reduction.
//!
//! This sample shows how to perform a reduction operation on an array of values to produce a
//! single value in a single kernel (as opposed to two or more kernel calls as shown in the
//! "reduction" CUDA Sample). Single-pass reduction requires a cooperative launch.
//!
//! Reductions are a very common computation in parallel algorithms. Any time an array of values
//! needs to be reduced to a single value using a binary associative operator, a reduction can be
//! used. Example applications include statistics computations such as mean and standard
//! deviation, and image processing applications such as finding the total luminance of an image.
//!
//! This code performs sum reductions, but any associative operator such as min() or max() could
//! also be used.
//!
//! It assumes the input size is a power of 2.
//!
//! Command line arguments:
//!
//! - `--n=<N>`: Specify the number of elements to reduce (default 33554432).
//! - `--threads=<N>`: Specify the number of threads per block (default 128).
//! - `--maxblocks=<N>`: Specify the maximum number of thread blocks to launch (kernel 6 only,
//!   default 64).

const std = @import("std");

const GridBarrier = std.nvptx.GridBarrier;

extern var sdata: [0]f64 align(16) addrspace(.shared);
var grid_counter: u32 addrspace(.global) = 0;

inline fn shflDown(value: f64, offset: u32, member_mask: std.nvptx.MemberMask) f64 {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.down, std.nvptx.warp_size);
    const words: [2]u32 = @bitCast(value);
    return @bitCast([2]u32{
        std.nvptx.shflSync(.down, u32, words[0], offset, clamp, member_mask),
        std.nvptx.shflSync(.down, u32, words[1], offset, clamp, member_mask),
    });
}

/// Parallel sum reduction using shared memory. It takes log(n) steps for n input elements, uses
/// n/2 work items and only works for power-of-2 arrays. This version adds multiple elements per
/// work item sequentially, which reduces the overall cost of the algorithm while keeping the
/// work complexity O(n) and the step complexity O(log n) (Brent's Theorem optimization).
inline fn reduceBlock(shared: [*]addrspace(.shared) f64) void {
    const tid = @workItemId(0);

    var sum = shared[tid];
    var offset: u32 = std.nvptx.warp_size / 2;
    while (offset > 0) : (offset /= 2) sum += shflDown(sum, offset, @fromBackingInt(std.nvptx.activemask()));
    shared[tid] = sum;
    std.nvptx.barrierSyncAligned(.work_group, null);

    if (tid == 0) {
        var beta: f64 = 0;
        var i: u32 = 0;
        while (i < @workGroupSize(0)) : (i += std.nvptx.warp_size) beta += shared[i];
        shared[0] = beta;
    }
    std.nvptx.barrierSyncAligned(.work_group, null);
}

/// Reduces an arbitrary size array in a single kernel invocation.
export fn reduceSinglePassMultiBlockCg(
    g_idata: [*]addrspace(.global) const f32,
    g_odata: [*]addrspace(.global) f32,
    n: u32,
) callconv(.kernel) void {
    const grid_barrier: GridBarrier = .{ .counter = &grid_counter };
    const shared: [*]addrspace(.shared) f64 = @ptrCast(&sdata);
    const block_thread_rank = @workItemId(0);
    const grid_size = std.nvptx.nctaid(0) * @workGroupSize(0);

    shared[block_thread_rank] = 0;

    var i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    while (i < n) : (i += grid_size) {
        shared[block_thread_rank] += g_idata[i];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    reduceBlock(shared);
    if (block_thread_rank == 0) {
        g_odata[@workGroupId(0)] = @floatCast(shared[0]);
    }
    grid_barrier.sync();

    if (@workGroupId(0) * @workGroupSize(0) + @workItemId(0) == 0) {
        for (1..std.nvptx.nctaid(0)) |block| {
            g_odata[0] += g_odata[block];
        }
    }
}
