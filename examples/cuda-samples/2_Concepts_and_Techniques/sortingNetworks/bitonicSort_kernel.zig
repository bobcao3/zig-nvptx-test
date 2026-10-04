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

//! Bitonic sort of (key, value) pairs. Based on
//! http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm

const std = @import("std");
const common = @import("sortingNetworks_common.zig");

const shared_size_limit = common.shared_size_limit;
const comparator = common.comparator;

const Global = [*]addrspace(.global) u32;

var s_key: [shared_size_limit]u32 addrspace(.shared) = undefined;
var s_val: [shared_size_limit]u32 addrspace(.shared) = undefined;

inline fn loadHalves(src_key: Global, src_val: Global, tid: u32) void {
    s_key[tid + 0] = src_key[0];
    s_val[tid + 0] = src_val[0];
    s_key[tid + (shared_size_limit / 2)] = src_key[(shared_size_limit / 2)];
    s_val[tid + (shared_size_limit / 2)] = src_val[(shared_size_limit / 2)];
}

inline fn storeHalves(dst_key: Global, dst_val: Global, tid: u32) void {
    dst_key[0] = s_key[tid + 0];
    dst_val[0] = s_val[tid + 0];
    dst_key[(shared_size_limit / 2)] = s_key[tid + (shared_size_limit / 2)];
    dst_val[(shared_size_limit / 2)] = s_val[tid + (shared_size_limit / 2)];
}

inline fn compareShared(tid: u32, stride: u32, dir: u32) void {
    const pos = 2 * tid - (tid & (stride - 1));
    comparator(&s_key[pos + 0], &s_val[pos + 0], &s_key[pos + stride], &s_val[pos + stride], dir);
}

/// Monolithic bitonic sort kernel for short arrays fitting into shared memory.
export fn bitonicSortShared(
    d_dst_key: Global,
    d_dst_val: Global,
    d_src_key: Global,
    d_src_val: Global,
    array_length: u32,
    dir: u32,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    const offset = @workGroupId(0) * shared_size_limit + tid;
    loadHalves(d_src_key + offset, d_src_val + offset, tid);

    var size: u32 = 2;
    while (size < array_length) : (size <<= 1) {
        const ddd = dir ^ @intFromBool((tid & (size / 2)) != 0);
        var stride = size / 2;
        while (stride > 0) : (stride >>= 1) {
            std.nvptx.barrierSyncAligned(.work_group, null);
            compareShared(tid, stride, ddd);
        }
    }

    // `ddd == dir` for the last bitonic merge step.
    var stride = array_length / 2;
    while (stride > 0) : (stride >>= 1) {
        std.nvptx.barrierSyncAligned(.work_group, null);
        compareShared(tid, stride, dir);
    }

    std.nvptx.barrierSyncAligned(.work_group, null);
    storeHalves(d_dst_key + offset, d_dst_val + offset, tid);
}

/// Bottom-level bitonic sort of the bitonic sort for large arrays (not fitting into shared
/// memory). Almost the same as `bitonicSortShared` with the exception of even and odd subarrays
/// being sorted in opposite directions. The bitonic merge accepts both orders.
export fn bitonicSortShared1(
    d_dst_key: Global,
    d_dst_val: Global,
    d_src_key: Global,
    d_src_val: Global,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    const offset = @workGroupId(0) * shared_size_limit + tid;
    loadHalves(d_src_key + offset, d_src_val + offset, tid);

    var size: u32 = 2;
    while (size < shared_size_limit) : (size <<= 1) {
        const ddd: u32 = @intFromBool((tid & (size / 2)) != 0);
        var stride = size / 2;
        while (stride > 0) : (stride >>= 1) {
            std.nvptx.barrierSyncAligned(.work_group, null);
            compareShared(tid, stride, ddd);
        }
    }

    // Odd and even arrays of `shared_size_limit` elements are sorted in opposite directions.
    const ddd = @workGroupId(0) & 1;
    var stride: u32 = shared_size_limit / 2;
    while (stride > 0) : (stride >>= 1) {
        std.nvptx.barrierSyncAligned(.work_group, null);
        compareShared(tid, stride, ddd);
    }

    std.nvptx.barrierSyncAligned(.work_group, null);
    storeHalves(d_dst_key + offset, d_dst_val + offset, tid);
}

/// Bitonic merge iteration for `stride >= shared_size_limit`.
export fn bitonicMergeGlobal(
    d_dst_key: Global,
    d_dst_val: Global,
    d_src_key: Global,
    d_src_val: Global,
    array_length: u32,
    size: u32,
    stride: u32,
    dir: u32,
) callconv(.kernel) void {
    const global_comparator_i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const comparator_i = global_comparator_i & (array_length / 2 - 1);

    const ddd = dir ^ @intFromBool((comparator_i & (size / 2)) != 0);
    const pos = 2 * global_comparator_i - (global_comparator_i & (stride - 1));

    var key_a = d_src_key[pos + 0];
    var val_a = d_src_val[pos + 0];
    var key_b = d_src_key[pos + stride];
    var val_b = d_src_val[pos + stride];

    comparator(&key_a, &val_a, &key_b, &val_b, ddd);

    d_dst_key[pos + 0] = key_a;
    d_dst_val[pos + 0] = val_a;
    d_dst_key[pos + stride] = key_b;
    d_dst_val[pos + stride] = val_b;
}

/// Combined bitonic merge steps for `size > shared_size_limit` and `stride` from
/// `shared_size_limit / 2` down to 1.
export fn bitonicMergeShared(
    d_dst_key: Global,
    d_dst_val: Global,
    d_src_key: Global,
    d_src_val: Global,
    array_length: u32,
    size: u32,
    dir: u32,
) callconv(.kernel) void {
    const tid = @workItemId(0);
    const offset = @workGroupId(0) * shared_size_limit + tid;
    loadHalves(d_src_key + offset, d_src_val + offset, tid);

    const comparator_i = common.umad(@workGroupId(0), @workGroupSize(0), tid) & ((array_length / 2) - 1);
    const ddd = dir ^ @intFromBool((comparator_i & (size / 2)) != 0);

    var stride: u32 = shared_size_limit / 2;
    while (stride > 0) : (stride >>= 1) {
        std.nvptx.barrierSyncAligned(.work_group, null);
        compareShared(tid, stride, ddd);
    }

    std.nvptx.barrierSyncAligned(.work_group, null);
    storeHalves(d_dst_key + offset, d_dst_val + offset, tid);
}
