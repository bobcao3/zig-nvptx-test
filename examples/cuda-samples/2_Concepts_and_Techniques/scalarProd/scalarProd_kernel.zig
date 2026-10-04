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

const accum_n = 1024;

var accum_result: [accum_n]f32 addrspace(.shared) = undefined;

/// Calculates the scalar products of `vector_n` pairs of vectors of `element_n` elements.
/// `element_n` should be a multiple of the warp size to meet the alignment constraints of memory
/// coalescing. `accum_n` must be a power of two.
export fn scalarProdGpu(
    d_c: [*]addrspace(.global) f32,
    d_a: [*]addrspace(.global) const f32,
    d_b: [*]addrspace(.global) const f32,
    vector_n: i32,
    element_n: i32,
) callconv(.kernel) void {
    const thread_x: i32 = @intCast(@workItemId(0));
    const block_x: i32 = @intCast(@workGroupId(0));
    const block_dim_x: i32 = @intCast(@workGroupSize(0));
    const grid_dim_x: i32 = @intCast(std.nvptx.nctaid(0));

    var vec = block_x;
    while (vec < vector_n) : (vec += grid_dim_x) {
        const vector_base = std.nvptx.math.mul24Lo(i32, element_n, vec);
        const vector_end = vector_base + element_n;

        var i_accum = thread_x;
        while (i_accum < accum_n) : (i_accum += block_dim_x) {
            var sum: f32 = 0;
            var pos = vector_base + i_accum;
            while (pos < vector_end) : (pos += accum_n) {
                sum += d_a[@intCast(pos)] * d_b[@intCast(pos)];
            }
            accum_result[@intCast(i_accum)] = sum;
        }

        var stride: i32 = accum_n / 2;
        while (stride > 0) : (stride >>= 1) {
            std.nvptx.barrierSyncAligned(.work_group, null);

            var i = thread_x;
            while (i < stride) : (i += block_dim_x) {
                accum_result[@intCast(i)] += accum_result[@intCast(stride + i)];
            }
        }

        std.nvptx.barrierSyncAligned(.work_group, null);

        if (thread_x == 0) d_c[@intCast(vec)] = accum_result[0];
    }
}
