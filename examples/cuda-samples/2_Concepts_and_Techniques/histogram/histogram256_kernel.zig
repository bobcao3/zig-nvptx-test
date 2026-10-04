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

//! 256-bin histogram of an 8-bit data array. Each warp counts into its own sub-histogram in
//! shared memory with atomic additions.

const std = @import("std");
const common = @import("histogram_common.zig");

const bin_count = common.histogram256_bin_count;
const work_group_size = common.histogram256_work_group_size;

var s_hist: [common.histogram256_work_group_memory]u32 addrspace(.shared) = undefined;
var s_merge: [common.merge_work_group_size]u32 addrspace(.shared) = undefined;

inline fn addByte(s_warp_hist: [*]addrspace(.shared) u32, data: u32) void {
    _ = @atomicRmw(u32, &s_warp_hist[data], .Add, 1, .monotonic);
}

inline fn addWord(s_warp_hist: [*]addrspace(.shared) u32, data: u32) void {
    inline for (.{ 0, 8, 16, 24 }) |shift| addByte(s_warp_hist, (data >> shift) & 0xFF);
}

/// Computes one partial histogram for each work group.
export fn histogram256Kernel(
    d_partial_histograms: [*]addrspace(.global) u32,
    d_data: [*]addrspace(.global) const u32,
    data_count: u32,
) callconv(.kernel) void {
    const thread_x = @workItemId(0);
    const s_warp_hist = @as([*]addrspace(.shared) u32, &s_hist) +
        (thread_x >> common.log2_warp_size) * bin_count;

    inline for (0..common.histogram256_work_group_memory / work_group_size) |i| {
        s_hist[thread_x + i * work_group_size] = 0;
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    var pos = @workGroupId(0) * @workGroupSize(0) + thread_x;
    while (pos < data_count) : (pos += @workGroupSize(0) * std.nvptx.nctaid(0)) {
        addWord(s_warp_hist, d_data[pos]);
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    var bin = thread_x;
    while (bin < bin_count) : (bin += work_group_size) {
        var sum: u32 = 0;
        for (0..common.warp_count) |i| {
            sum += s_hist[bin + i * bin_count];
        }
        d_partial_histograms[@workGroupId(0) * bin_count + bin] = sum;
    }
}

/// Adds up the same bin of every partial histogram, with one work group for each bin.
export fn mergeHistogram256Kernel(
    d_histogram: [*]addrspace(.global) u32,
    d_partial_histograms: [*]addrspace(.global) const u32,
    histogram_count: u32,
) callconv(.kernel) void {
    const thread_x = @workItemId(0);
    const merge_size = common.merge_work_group_size;

    var sum: u32 = 0;
    var i = thread_x;
    while (i < histogram_count) : (i += merge_size) {
        sum += d_partial_histograms[@workGroupId(0) + i * bin_count];
    }

    s_merge[thread_x] = sum;

    var stride: u32 = merge_size / 2;
    while (stride > 0) : (stride >>= 1) {
        std.nvptx.barrierSyncAligned(.work_group, null);

        if (thread_x < stride) {
            s_merge[thread_x] += s_merge[thread_x + stride];
        }
    }

    if (thread_x == 0) {
        d_histogram[@workGroupId(0)] = s_merge[0];
    }
}
