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

//! Shuffle intrinsics CUDA Sample
//! This sample demonstrates the use of the shuffle intrinsic.
//! First, a simple example of a prefix sum using the shuffle to
//! perform a scan operation is provided.
//! Secondly, a more involved example of computing an integral image
//! using the shuffle intrinsic is provided, where the shuffle
//! scan operation and shuffle xor operations are used.

const std = @import("std");
const integral_image = @import("shfl_integral_image.zig");

extern var sums: [0]i32 align(16) addrspace(.shared);
var buf: i32 addrspace(.shared) = undefined;

/// Scan using shfl - takes log2(n) steps.
/// This function demonstrates basic use of the shuffle intrinsic, shfl_up,
/// to perform a scan operation across a work group.
/// First, it performs a scan (prefix sum in this case) inside a warp.
/// Then to continue the scan operation across the work group,
/// each warp's sum is placed into shared memory. A single warp
/// then performs a shuffle scan on that shared memory. The results
/// are then uniformly added to each warp's work items.
/// This pyramid type approach is continued by placing each work group's
/// final sum in global memory and prefix summing that via another kernel call,
/// then uniformly adding across the input data via the `uniformAdd` kernel.
fn shflScanTest(
    data: [*]addrspace(.global) i32,
    width: u32,
    partial_sums: ?[*]addrspace(.global) i32,
) callconv(.kernel) void {
    const warp_sums: [*]addrspace(.shared) i32 = @ptrCast(&sums);
    const work_item = @workItemId(0);
    const id = @workGroupId(0) * @workGroupSize(0) + work_item;
    const lane_id = id % std.nvptx.warp_size;
    const warp_id = work_item / std.nvptx.warp_size;

    var value = data[id];

    var i: u32 = 1;
    while (i <= width) : (i *= 2) {
        const n = std.nvptx.shflSync(.up, i32, value, i, .fromWidth(.up, width), .all);

        if (lane_id >= i) value +%= n;
    }

    if (work_item % std.nvptx.warp_size == std.nvptx.warp_size - 1) {
        warp_sums[warp_id] = value;
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    const warps = @workGroupSize(0) / std.nvptx.warp_size;
    if (warp_id == 0 and lane_id < warps) {
        var warp_sum = warp_sums[lane_id];

        const mask = (@as(u32, 1) << @intCast(warps)) - 1;
        var j: u32 = 1;
        while (j <= warps) : (j *= 2) {
            const n = std.nvptx.shflSync(.up, i32, warp_sum, j, .fromWidth(.up, warps), @fromBackingInt(mask));

            if (lane_id >= j) warp_sum +%= n;
        }

        warp_sums[lane_id] = warp_sum;
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    var block_sum: i32 = 0;

    if (warp_id > 0) block_sum = warp_sums[warp_id - 1];

    value +%= block_sum;

    data[id] = value;

    if (partial_sums) |partial| {
        if (work_item == @workGroupSize(0) - 1) partial[@workGroupId(0)] = value;
    }
}

/// Uniform add: add partial sums array.
fn uniformAdd(
    data: [*]addrspace(.global) i32,
    partial_sums: [*]addrspace(.global) i32,
    len: u32,
) callconv(.kernel) void {
    const id = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    if (id > len) return;

    if (@workItemId(0) == 0) buf = partial_sums[@workGroupId(0)];

    std.nvptx.barrierSyncAligned(.work_group, null);
    data[id] +%= buf;
}

comptime {
    @export(&shflScanTest, .{ .name = "shflScanTest" });
    @export(&uniformAdd, .{ .name = "uniformAdd" });
    @export(&integral_image.shflIntimageRows, .{ .name = "shflIntimageRows" });
    @export(&integral_image.shflVerticalShfl, .{ .name = "shflVerticalShfl" });
}
