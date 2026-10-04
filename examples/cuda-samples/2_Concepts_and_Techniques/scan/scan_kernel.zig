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

const std = @import("std");
const common = @import("scan_common.zig");

const threadblock_size = common.threadblock_size;
const Uint4 = @Vector(4, u32);
const SharedWords = [*]addrspace(.shared) volatile u32;

var s_data: [2 * threadblock_size]u32 addrspace(.shared) = undefined;
var s_data2: [2 * threadblock_size]u32 addrspace(.shared) = undefined;
var buf: u32 addrspace(.shared) = undefined;

/// Naive inclusive scan: O(N * log2(N)) operations. `s_data` needs 2 * `size` words. The first
/// half holds `size` zeros, which avoids the `pos >= offset` test and saves instructions.
inline fn scan1Inclusive(idata: u32, s_data_words: SharedWords, size: u32) u32 {
    const thread_x = @workItemId(0);
    var pos = 2 * thread_x - (thread_x & (size - 1));
    s_data_words[pos] = 0;
    pos += size;
    s_data_words[pos] = idata;

    var offset: u32 = 1;
    while (offset < size) : (offset <<= 1) {
        std.nvptx.barrierSyncAligned(.work_group, null);
        const t = s_data_words[pos] +% s_data_words[pos - offset];
        std.nvptx.barrierSyncAligned(.work_group, null);
        s_data_words[pos] = t;
    }

    return s_data_words[pos];
}

inline fn scan1Exclusive(idata: u32, s_data_words: SharedWords, size: u32) u32 {
    return scan1Inclusive(idata, s_data_words, size) -% idata;
}

inline fn scan4Inclusive(idata4_in: Uint4, s_data_words: SharedWords, size: u32) Uint4 {
    var idata4 = idata4_in;
    idata4[1] +%= idata4[0];
    idata4[2] +%= idata4[1];
    idata4[3] +%= idata4[2];

    const oval = scan1Exclusive(idata4[3], s_data_words, size / 4);

    idata4 +%= @splat(oval);
    return idata4;
}

/// Exclusive vector scan: the array to be scanned is stored in the registers of the work item as
/// a `Uint4`.
inline fn scan4Exclusive(idata4: Uint4, s_data_words: SharedWords, size: u32) Uint4 {
    return scan4Inclusive(idata4, s_data_words, size) -% idata4;
}

export fn scanExclusiveShared(
    d_dst: [*]addrspace(.global) Uint4,
    d_src: [*]addrspace(.global) Uint4,
    size: u32,
) callconv(.kernel) void {
    const pos = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    const idata4 = d_src[pos];
    const odata4 = scan4Exclusive(idata4, &s_data, size);
    d_dst[pos] = odata4;
}

/// Exclusive scan of the top elements of the bottom-level scans (4 * `threadblock_size`).
export fn scanExclusiveShared2(
    d_buf: [*]addrspace(.global) u32,
    d_dst: [*]addrspace(.global) u32,
    d_src: [*]addrspace(.global) u32,
    n: u32,
    array_length: u32,
) callconv(.kernel) void {
    const pos = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    var idata: u32 = 0;
    if (pos < n) {
        const top = (4 * threadblock_size) - 1 + (4 * threadblock_size) * pos;
        idata = d_dst[top] +% d_src[top];
    }

    const odata = scan1Exclusive(idata, &s_data2, array_length);

    if (pos < n) {
        d_buf[pos] = odata;
    }
}

/// The final step of the large-array scan: combines the basic inclusive scan with the exclusive
/// scan of the top elements of the input arrays.
export fn uniformUpdate(d_data: [*]addrspace(.global) Uint4, d_buffer: [*]addrspace(.global) u32) callconv(.kernel) void {
    const pos = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    if (@workItemId(0) == 0) {
        buf = d_buffer[@workGroupId(0)];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    var data4 = d_data[pos];
    data4 +%= @splat(buf);
    d_data[pos] = data4;
}
