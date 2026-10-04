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

//! Parallel reduction kernels.

const std = @import("std");
const common = @import("threadFenceReduction_common.zig");

extern var shared_memory: [0]f32 align(16) addrspace(.shared);

/// The number of work items that each reduction step handles.
const vec = 32;

export var retirement_count: u32 addrspace(.global) = 0;

inline fn syncTile(comptime block_size: u32) void {
    const tile_mask: u32 = if (block_size < vec) (1 << block_size) - 1 else 0xffffffff;
    std.nvptx.barWarpSync(@fromBackingInt(tile_mask));
}

/// Parallel sum reduction using shared memory. It takes log(n) steps for n input elements, uses
/// n/2 work items and only works for power-of-2 arrays. This version adds multiple elements per
/// work item sequentially, which reduces the overall cost of the algorithm while keeping the
/// work complexity O(n) and the step complexity O(log n) (Brent's Theorem optimization).
inline fn reduceBlock(
    comptime block_size: u32,
    sdata: [*]addrspace(.shared) volatile f32,
    my_sum: f32,
    tid: u32,
) void {
    sdata[tid] = my_sum;
    syncTile(block_size);

    const vid = tid & (vec - 1);

    var beta = my_sum;
    var temp: f32 = undefined;

    var i: u32 = vec / 2;
    while (i > 0) : (i >>= 1) {
        if (vid < i) {
            temp = sdata[tid + i];
            beta += temp;
            sdata[tid] = beta;
        }
        syncTile(block_size);
    }
    std.nvptx.barrierSyncAligned(.work_group, null);

    if (tid == 0) {
        beta = 0;
        i = 0;
        while (i < @workGroupSize(0)) : (i += vec) {
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    std.nvptx.barrierSyncAligned(.work_group, null);
}

inline fn reduceBlocks(
    comptime block_size: u32,
    comptime n_is_pow2: bool,
    g_idata: [*]addrspace(.global) const f32,
    g_odata: [*]addrspace(.global) f32,
    n: u32,
) void {
    const sdata: [*]addrspace(.shared) volatile f32 = @ptrCast(&shared_memory);

    const tid = @workItemId(0);
    var i = @workGroupId(0) * (block_size * 2) + @workItemId(0);
    const grid_size = block_size * 2 * std.nvptx.nctaid(0);
    var my_sum: f32 = 0;

    while (i < n) {
        my_sum += g_idata[i];

        if (n_is_pow2 or i + block_size < n) my_sum += g_idata[i + block_size];

        i += grid_size;
    }

    reduceBlock(block_size, sdata, my_sum, tid);

    if (tid == 0) g_odata[@workGroupId(0)] = sdata[0];
}

fn reduceMultiPass(comptime block_size: u32, comptime n_is_pow2: bool) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const f32,
            g_odata: [*]addrspace(.global) f32,
            n: u32,
        ) callconv(.kernel) void {
            reduceBlocks(block_size, n_is_pow2, g_idata, g_odata, n);
        }

        comptime {
            const name = common.kernelName("reduceMultiPass", block_size, n_is_pow2);
            @export(&kernel, .{ .name = name });
        }
    };
}

/// Reduces an arbitrary size array in a single kernel invocation. It does so by keeping track of
/// how many work groups have finished. After each work group completes the reduction of its own
/// block of data, it takes a ticket by atomically incrementing a global counter. If the ticket
/// value is equal to the number of work groups, then the work group holding the ticket knows
/// that it is the last one to finish. This last work group is responsible for summing the
/// results of all the other work groups.
///
/// In order for this to work, all memory transactions of a work group must have completed before
/// it takes a ticket. The fence blocks until the results of all outstanding memory transactions
/// within the calling work item are visible to all other work items.
fn reduceSinglePass(comptime block_size: u32, comptime n_is_pow2: bool) type {
    return struct {
        var am_last: bool addrspace(.shared) = undefined;

        fn kernel(
            g_idata: [*]addrspace(.global) const f32,
            g_odata: [*]addrspace(.global) f32,
            n: u32,
        ) callconv(.kernel) void {
            reduceBlocks(block_size, n_is_pow2, g_idata, g_odata, n);

            if (std.nvptx.nctaid(0) > 1) {
                const tid = @workItemId(0);
                const smem: [*]addrspace(.shared) volatile f32 = @ptrCast(&shared_memory);

                std.nvptx.fence(.seq_cst, .gpu);

                if (tid == 0) {
                    const grid_dim = std.nvptx.nctaid(0);
                    const ticket = std.nvptx.atom(
                        .gpu,
                        .monotonic,
                        .inc,
                        &retirement_count,
                        grid_dim,
                    );
                    am_last = ticket == grid_dim - 1;
                }

                std.nvptx.barrierSyncAligned(.work_group, null);

                if (am_last) {
                    var i = tid;
                    var my_sum: f32 = 0;

                    while (i < std.nvptx.nctaid(0)) {
                        my_sum += g_odata[i];
                        i += block_size;
                    }

                    reduceBlock(block_size, smem, my_sum, tid);

                    if (tid == 0) {
                        g_odata[0] = smem[0];

                        retirement_count = 0;
                    }
                }
            }
        }

        comptime {
            const name = common.kernelName("reduceSinglePass", block_size, n_is_pow2);
            @export(&kernel, .{ .name = name });
        }
    };
}

comptime {
    for (common.block_sizes) |block_size| {
        _ = reduceMultiPass(block_size, true);
        _ = reduceMultiPass(block_size, false);
        _ = reduceSinglePass(block_size, true);
        _ = reduceSinglePass(block_size, false);
    }
}
