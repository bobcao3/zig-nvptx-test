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

//! 64-bin histogram of an 8-bit data array. Each work item counts into its own byte-sized
//! counters in shared memory.

const std = @import("std");
const common = @import("histogram_common.zig");

const bin_count = common.histogram64_bin_count;
const work_group_size = common.histogram64_work_group_size;
const banks = common.shared_memory_banks;

const Data = @Vector(4, u32);

var s_hist: [work_group_size * bin_count]u8 align(4) addrspace(.shared) = undefined;
var s_merge: [common.merge_work_group_size]u32 addrspace(.shared) = undefined;

inline fn addByte(s_thread_base: [*]addrspace(.shared) u8, data: u32) void {
    s_thread_base[data * work_group_size] += 1;
}

inline fn addWord(s_thread_base: [*]addrspace(.shared) u8, data: u32) void {
    inline for (.{ 2, 10, 18, 26 }) |shift| addByte(s_thread_base, (data >> shift) & 0x3F);
}

/// Computes one partial histogram for each work group.
export fn histogram64Kernel(
    d_partial_histograms: [*]addrspace(.global) u32,
    d_data: [*]align(16) addrspace(.global) const Data,
    data_count: u32,
) callconv(.kernel) void {
    const thread_x = @workItemId(0);
    const thread_pos = (thread_x & ~@as(u32, banks * 4 - 1)) |
        ((thread_x & (banks - 1)) << 2) |
        ((thread_x & (banks * 3)) >> 4);

    const s_thread_base: [*]addrspace(.shared) u8 = @as([*]addrspace(.shared) u8, &s_hist) + thread_pos;
    const s_words: [*]addrspace(.shared) u32 = @ptrCast(&s_hist);

    inline for (0..bin_count / 4) |i| {
        s_words[thread_x + i * work_group_size] = 0;
    }

    // Each counter is a byte, so a work item can submit at most 255 values.
    std.nvptx.barrierSyncAligned(.work_group, null);

    var pos = @workGroupId(0) * @workGroupSize(0) + thread_x;
    while (pos < data_count) : (pos += @workGroupSize(0) * std.nvptx.nctaid(0)) {
        const data = d_data[pos];
        addWord(s_thread_base, data[0]);
        addWord(s_thread_base, data[1]);
        addWord(s_thread_base, data[2]);
        addWord(s_thread_base, data[3]);
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    if (thread_x < bin_count) {
        const s_hist_base = @as([*]addrspace(.shared) u8, &s_hist) + thread_x * work_group_size;

        var sum: u32 = 0;
        var i_pos: u32 = 4 * (thread_x & (banks - 1));

        inline for (0..work_group_size / 4) |_| {
            sum += @as(u32, s_hist_base[i_pos + 0]) + s_hist_base[i_pos + 1] +
                s_hist_base[i_pos + 2] + s_hist_base[i_pos + 3];
            i_pos = (i_pos + 4) & (work_group_size - 1);
        }

        d_partial_histograms[@workGroupId(0) * bin_count + thread_x] = sum;
    }
}

/// Adds up the same bin of every partial histogram, with one work group for each bin.
export fn mergeHistogram64Kernel(
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
