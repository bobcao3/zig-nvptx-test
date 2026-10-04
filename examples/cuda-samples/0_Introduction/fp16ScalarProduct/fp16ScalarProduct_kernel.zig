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

//! Scalar product of two half-precision vectors with `f16` pairs, computed once with the
//! arithmetic operators and once with fused multiply-add and add on pairs.

const std = @import("std");

const Half2 = @Vector(2, f16);

const num_of_threads = 128;

var native_array: [num_of_threads]Half2 addrspace(.shared) = undefined;
var intrinsics_array: [num_of_threads]Half2 addrspace(.shared) = undefined;

inline fn reduceInSharedIntrinsics(v: *addrspace(.shared) [num_of_threads]Half2) void {
    const thread_index = @workItemId(0);
    comptime var offset = num_of_threads / 2;
    inline while (offset >= 1) : (offset /= 2) {
        if (thread_index < offset) v[thread_index] = v[thread_index] + v[thread_index + offset];
        std.nvptx.barrierSyncAligned(.work_group, null);
    }
}

inline fn reduceInSharedNative(v: *addrspace(.shared) [num_of_threads]Half2) void {
    const thread_index = @workItemId(0);
    comptime var offset = num_of_threads / 2;
    inline while (offset >= 1) : (offset /= 2) {
        if (thread_index < offset) v[thread_index] = v[thread_index] + v[thread_index + offset];
        std.nvptx.barrierSyncAligned(.work_group, null);
    }
}

export fn scalarProductKernelIntrinsics(
    a: [*]addrspace(.global) const Half2,
    b: [*]addrspace(.global) const Half2,
    results: [*]addrspace(.global) f32,
    size: usize,
) callconv(.kernel) void {
    const stride = std.nvptx.nctaid(0) * @workGroupSize(0);
    const thread_index = @workItemId(0);

    intrinsics_array[thread_index] = @splat(0);
    var value: Half2 = @splat(0);

    var i = thread_index + @workGroupSize(0) * @workGroupId(0);
    while (i < size) : (i += stride) {
        value = @mulAdd(Half2, a[i], b[i], value);
    }

    intrinsics_array[thread_index] = value;
    std.nvptx.barrierSyncAligned(.work_group, null);
    reduceInSharedIntrinsics(&intrinsics_array);

    if (thread_index == 0) {
        const result = intrinsics_array[0];
        const f_result = @as(f32, result[0]) + @as(f32, result[1]);
        results[@workGroupId(0)] = f_result;
    }
}

export fn scalarProductKernelNative(
    a: [*]addrspace(.global) const Half2,
    b: [*]addrspace(.global) const Half2,
    results: [*]addrspace(.global) f32,
    size: usize,
) callconv(.kernel) void {
    const stride = std.nvptx.nctaid(0) * @workGroupSize(0);
    const thread_index = @workItemId(0);

    var value: Half2 = @splat(0);
    native_array[thread_index] = value;

    var i = thread_index + @workGroupSize(0) * @workGroupId(0);
    while (i < size) : (i += stride) {
        value = a[i] * b[i] + value;
    }

    native_array[thread_index] = value;
    std.nvptx.barrierSyncAligned(.work_group, null);
    reduceInSharedNative(&native_array);

    if (thread_index == 0) {
        const result = native_array[0];
        const f_result = @as(f32, result[1]) + @as(f32, result[0]);
        results[@workGroupId(0)] = f_result;
    }
}
