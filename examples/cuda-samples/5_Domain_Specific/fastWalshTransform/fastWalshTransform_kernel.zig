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

//! Walsh transforms belong to a class of generalized Fourier transformations. They have
//! applications in various fields of electrical engineering and numeric theory. In this sample we
//! demonstrate efficient implementation of naturally-ordered Walsh transform (also known as
//! Walsh-Hadamard or Hadamard transform) in CUDA and its particular application to dyadic
//! convolution computation. Refer to excellent Jorg Arndt's "Algorithms for Programmers"
//! textbook http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
//!
//! Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)

const std = @import("std");

extern var s_data: [0]f32 align(16) addrspace(.shared);

/// Elementary (for vectors less than elementary size) in-shared memory combined radix-2 + radix-4
/// Fast Walsh Transform.
export fn fwtBatch1Kernel(
    d_output: [*]addrspace(.global) f32,
    d_input: [*]addrspace(.global) const f32,
    log2n: u32,
) callconv(.kernel) void {
    const n = @as(u32, 1) << @intCast(log2n);
    const base = @workGroupId(0) << @intCast(log2n);

    // (2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    const shared: [*]addrspace(.shared) f32 = &s_data;
    const d_src = d_input + base;
    const d_dst = d_output + base;

    var pos = @workItemId(0);
    while (pos < n) : (pos += @workGroupSize(0)) {
        shared[pos] = d_src[pos];
    }

    // Main radix-4 stages
    pos = @workItemId(0);

    var stride = n >> 2;
    while (stride > 0) : (stride >>= 2) {
        const lo = pos & (stride - 1);
        const index0 = ((pos - lo) << 2) + lo;
        const index1 = index0 + stride;
        const index2 = index1 + stride;
        const index3 = index2 + stride;

        std.nvptx.barrierSyncAligned(.work_group, null);
        var d0 = shared[index0];
        var d1 = shared[index1];
        var d2 = shared[index2];
        var d3 = shared[index3];

        var t = d0;
        d0 = d0 + d2;
        d2 = t - d2;
        t = d1;
        d1 = d1 + d3;
        d3 = t - d3;
        t = d0;
        shared[index0] = d0 + d1;
        shared[index1] = t - d1;
        t = d2;
        shared[index2] = d2 + d3;
        shared[index3] = t - d3;
    }

    // Do single radix-2 stage for odd power of two
    if (log2n & 1 != 0) {
        std.nvptx.barrierSyncAligned(.work_group, null);

        var radix2_pos = @workItemId(0);
        while (radix2_pos < n / 2) : (radix2_pos += @workGroupSize(0)) {
            const index0 = radix2_pos << 1;
            const index1 = index0 + 1;

            const d0 = shared[index0];
            const d1 = shared[index1];
            shared[index0] = d0 + d1;
            shared[index1] = d0 - d1;
        }
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    var store_pos = @workItemId(0);
    while (store_pos < n) : (store_pos += @workGroupSize(0)) {
        d_dst[store_pos] = shared[store_pos];
    }
}

/// Single in-global memory radix-4 Fast Walsh Transform pass (for strides exceeding elementary
/// vector size).
export fn fwtBatch2Kernel(
    d_output: [*]addrspace(.global) f32,
    d_input: [*]addrspace(.global) const f32,
    stride: u32,
) callconv(.kernel) void {
    const pos = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const n = @workGroupSize(0) * std.nvptx.nctaid(0) * 4;

    const d_src = d_input + @workGroupId(1) * n;
    const d_dst = d_output + @workGroupId(1) * n;

    const lo = pos & (stride - 1);
    const index0 = ((pos - lo) << 2) + lo;
    const index1 = index0 + stride;
    const index2 = index1 + stride;
    const index3 = index2 + stride;

    var d0 = d_src[index0];
    var d1 = d_src[index1];
    var d2 = d_src[index2];
    var d3 = d_src[index3];

    var t = d0;
    d0 = d0 + d2;
    d2 = t - d2;
    t = d1;
    d1 = d1 + d3;
    d3 = t - d3;
    t = d0;
    d_dst[index0] = d0 + d1;
    d_dst[index1] = t - d1;
    t = d2;
    d_dst[index2] = d2 + d3;
    d_dst[index3] = t - d3;
}

/// Modulate two arrays.
export fn modulateKernel(
    d_a: [*]addrspace(.global) f32,
    d_b: [*]addrspace(.global) const f32,
    n: u32,
) callconv(.kernel) void {
    const tid = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const num_threads = @workGroupSize(0) * std.nvptx.nctaid(0);
    const rcp_n = 1.0 / @as(f32, @floatFromInt(n));

    var pos = tid;
    while (pos < n) : (pos += num_threads) {
        d_a[pos] *= d_b[pos] * rcp_n;
    }
}
