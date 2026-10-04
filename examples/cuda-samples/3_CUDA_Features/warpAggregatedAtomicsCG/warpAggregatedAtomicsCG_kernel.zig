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

//! This sample demonstrates how to use cooperative groups to perform warp-aggregated atomics,
//! a useful technique to improve performance when many work items atomically add to a single
//! counter. In this sample, an input array is filtered so that only the elements greater than 0
//! are copied to the output array. It also demonstrates the warp-aggregated atomic increment and
//! maximum for multiple buckets, which use labeled partitions of coalesced groups.

const std = @import("std");

const Group = struct {
    mask: u32,

    inline fn threadRank(group: Group) u32 {
        return @popCount(group.mask & std.nvptx.lanemaskLt());
    }

    inline fn size(group: Group) u32 {
        return @popCount(group.mask);
    }

    inline fn shflFromLeader(group: Group, value: i32) i32 {
        const leader = @ctz(group.mask);
        const clamp: std.nvptx.ShuffleClamp = .fromWidth(.idx, std.nvptx.warp_size);
        return std.nvptx.shflSync(.idx, i32, value, leader, clamp, @fromBackingInt(group.mask));
    }

    inline fn coalescedThreads() Group {
        return .{ .mask = std.nvptx.activemask() };
    }

    inline fn labeledPartition(active: Group, label: i32) Group {
        return .{ .mask = std.nvptx.matchAnySync(i32, label, @fromBackingInt(active.mask)) };
    }
};

inline fn atomicAggInc(counter: *addrspace(.global) i32) i32 {
    const active: Group = .coalescedThreads();

    var res: i32 = 0;
    if (active.threadRank() == 0) {
        res = @atomicRmw(i32, counter, .Add, @intCast(active.size()), .monotonic);
    }

    res = active.shflFromLeader(res);

    return res + @as(i32, @intCast(active.threadRank()));
}

export fn filterArr(
    dst: [*]addrspace(.global) i32,
    nres: *addrspace(.global) i32,
    src: [*]addrspace(.global) const i32,
    n: i32,
) callconv(.kernel) void {
    const id = @workItemId(0) + @workGroupId(0) * @workGroupSize(0);

    var i = id;
    while (i < n) : (i += std.nvptx.nctaid(0) * @workGroupSize(0)) {
        if (src[i] > 0) dst[@intCast(atomicAggInc(nres))] = src[i];
    }
}

inline fn atomicAggIncMulti(bucket: i32, counter: [*]addrspace(.global) i32) i32 {
    const active: Group = .coalescedThreads();
    const labeled_group = active.labeledPartition(bucket);

    var res: i32 = 0;
    if (labeled_group.threadRank() == 0) {
        const increment: i32 = @intCast(labeled_group.size());
        res = @atomicRmw(i32, &counter[@intCast(bucket)], .Add, increment, .monotonic);
    }

    res = labeled_group.shflFromLeader(res);

    return res + @as(i32, @intCast(labeled_group.threadRank()));
}

export fn mapToBuckets(
    src_arr: [*]addrspace(.global) const i32,
    indices_buckets: [*]addrspace(.global) i32,
    bucket_counters: [*]addrspace(.global) i32,
    src_size: i32,
    num_of_buckets: i32,
) callconv(.kernel) void {
    const grid_size = std.nvptx.nctaid(0) * @workGroupSize(0);

    var i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    while (i < src_size) : (i += grid_size) {
        const bucket = src_arr[i];
        if (bucket < num_of_buckets) {
            indices_buckets[@intCast(atomicAggIncMulti(bucket, bucket_counters))] = @intCast(i);
        }
    }
}

inline fn atomicAggMaxMulti(
    bucket: i32,
    counter: [*]addrspace(.global) i32,
    value_for_max: i32,
) void {
    const active: Group = .coalescedThreads();
    const labeled_group = active.labeledPartition(bucket);

    const max_value_in_group = std.nvptx.reduxSync(.max, i32, value_for_max, @fromBackingInt(labeled_group.mask));

    if (labeled_group.threadRank() == 0) {
        _ = @atomicRmw(i32, &counter[@intCast(bucket)], .Max, max_value_in_group, .monotonic);
    }
}

export fn calculateMaxInEachBuckets(
    src_arr: [*]addrspace(.global) const i32,
    value_in_buckets: [*]addrspace(.global) const i32,
    buckets_max: [*]addrspace(.global) i32,
    src_size: i32,
    num_of_buckets: i32,
) callconv(.kernel) void {
    const grid_size = std.nvptx.nctaid(0) * @workGroupSize(0);

    var i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    while (i < src_size) : (i += grid_size) {
        const bucket = src_arr[i];
        if (bucket < num_of_buckets) {
            atomicAggMaxMulti(bucket, buckets_max, value_in_buckets[i]);
        }
    }
}
