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

//! Odd-even merge sort (Batcher's sort) of (key, value) pairs. Based on
//! http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm

const std = @import("std");
const common = @import("sortingNetworks_common.zig");

const shared_size_limit = common.shared_size_limit;
const comparator = common.comparator;

const Global = [*]addrspace(.global) u32;

var s_key: [shared_size_limit]u32 addrspace(.shared) = undefined;
var s_val: [shared_size_limit]u32 addrspace(.shared) = undefined;

/// Monolithic Batcher's sort kernel for short arrays fitting into shared memory.
export fn oddEvenMergeSortShared(
    d_dst_key: Global,
    d_dst_val: Global,
    d_src_key: Global,
    d_src_val: Global,
    array_length: u32,
    dir: u32,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    const offset_global = @workGroupId(0) * shared_size_limit + tid;
    const src_key = d_src_key + offset_global;
    const src_val = d_src_val + offset_global;
    const dst_key = d_dst_key + offset_global;
    const dst_val = d_dst_val + offset_global;
    s_key[tid + 0] = src_key[0];
    s_val[tid + 0] = src_val[0];
    s_key[tid + (shared_size_limit / 2)] = src_key[(shared_size_limit / 2)];
    s_val[tid + (shared_size_limit / 2)] = src_val[(shared_size_limit / 2)];

    var size: u32 = 2;
    while (size <= array_length) : (size <<= 1) {
        var stride = size / 2;
        const offset = tid & (stride - 1);

        {
            std.nvptx.barrierSyncAligned(.work_group, null);
            const pos = 2 * tid - (tid & (stride - 1));
            comparator(&s_key[pos + 0], &s_val[pos + 0], &s_key[pos + stride], &s_val[pos + stride], dir);
            stride >>= 1;
        }

        while (stride > 0) : (stride >>= 1) {
            std.nvptx.barrierSyncAligned(.work_group, null);
            const pos = 2 * tid - (tid & (stride - 1));
            if (offset >= stride) {
                comparator(&s_key[pos - stride], &s_val[pos - stride], &s_key[pos + 0], &s_val[pos + 0], dir);
            }
        }
    }

    std.nvptx.barrierSyncAligned(.work_group, null);
    dst_key[0] = s_key[tid + 0];
    dst_val[0] = s_val[tid + 0];
    dst_key[(shared_size_limit / 2)] = s_key[tid + (shared_size_limit / 2)];
    dst_val[(shared_size_limit / 2)] = s_val[tid + (shared_size_limit / 2)];
}

/// Odd-even merge sort iteration kernel for large arrays (not fitting into shared memory).
export fn oddEvenMergeGlobal(
    d_dst_key: Global,
    d_dst_val: Global,
    d_src_key: Global,
    d_src_val: Global,
    array_length: u32,
    size: u32,
    stride: u32,
    dir: u32,
) callconv(.kernel) void {
    _ = array_length;
    const global_comparator_i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    const pos = 2 * global_comparator_i - (global_comparator_i & (stride - 1));

    if (stride < size / 2) {
        const offset = global_comparator_i & ((size / 2) - 1);

        if (offset >= stride) {
            var key_a = d_src_key[pos - stride];
            var val_a = d_src_val[pos - stride];
            var key_b = d_src_key[pos + 0];
            var val_b = d_src_val[pos + 0];

            comparator(&key_a, &val_a, &key_b, &val_b, dir);

            d_dst_key[pos - stride] = key_a;
            d_dst_val[pos - stride] = val_a;
            d_dst_key[pos + 0] = key_b;
            d_dst_val[pos + 0] = val_b;
        }
    } else {
        var key_a = d_src_key[pos + 0];
        var val_a = d_src_val[pos + 0];
        var key_b = d_src_key[pos + stride];
        var val_b = d_src_val[pos + stride];

        comparator(&key_a, &val_a, &key_b, &val_b, dir);

        d_dst_key[pos + 0] = key_a;
        d_dst_val[pos + 0] = val_a;
        d_dst_key[pos + stride] = key_b;
        d_dst_val[pos + stride] = val_b;
    }
}
