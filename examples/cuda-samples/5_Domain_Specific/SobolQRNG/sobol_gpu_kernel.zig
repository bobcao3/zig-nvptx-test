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
//
// Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights
// reserved.
// Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights
// reserved.

//! Sobol Quasi-random Number Generator example
//!
//! Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
//! http://people.maths.ox.ac.uk/~gilesm/
//!
//! and C code developed by Stephen Joe, University of Waikato, New Zealand
//! and Frances Kuo, University of New South Wales, Australia
//! http://web.maths.unsw.edu.au/~fkuo/sobol/
//!
//! For theoretical background see:
//!
//! P. Bratley and B.L. Fox.
//! Implementing Sobol's quasirandom sequence generator
//! http://portal.acm.org/citation.cfm?id=42288
//! ACM Trans. on Math. Software, 14(1):88-100, 1988
//!
//! S. Joe and F. Kuo.
//! Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
//! http://portal.acm.org/citation.cfm?id=641879
//! ACM Trans. on Math. Software, 29(1):49-57, 2003

const std = @import("std");
const sobol = @import("sobol.zig");

const n_directions = sobol.n_directions;
const ffs = sobol.ffs;

const k_2powneg32: f32 = 2.3283064E-10;

var v: [n_directions]u32 addrspace(.shared) = undefined;

export fn sobolGpuKernel(
    n_vectors: u32,
    n_dimensions: u32,
    d_directions: [*]addrspace(.global) const u32,
    d_output: [*]addrspace(.global) f32,
) callconv(.kernel) void {
    _ = n_dimensions;

    // Offsets into the dimension that the second work group index specifies.
    const directions = d_directions + n_directions * @workGroupId(1);
    const output = d_output + n_vectors * @workGroupId(1);

    // Copies the direction numbers of this dimension into shared memory. There are only 32
    // direction numbers, so only the first 32 work items take part.
    if (@workItemId(0) < n_directions) {
        v[@workItemId(0)] = directions[@workItemId(0)];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    // The index of the vector that this work item computes first, and the step to the next
    // vector of this work item.
    const first_index = @workItemId(0) + @workGroupId(0) * @workGroupSize(0);
    const stride = std.nvptx.nctaid(0) * @workGroupSize(0);

    // The Gray code of the index. See Numerical Recipes in C, chapter 20.
    var g = first_index ^ (first_index >> 1);

    // Initializes the first point x[first_index]. This is equation (*) in the paper of Bratley
    // and Fox, which computes the value of x[n] without knowing x[n-1].
    var x: u32 = 0;

    var k: u32 = 0;
    while (k < ffs(stride) - 1) : (k += 1) {
        // Computes x ^= g_k * v[k], where g_k is one or zero, with a mask whose bits all equal
        // g_k. The shift of g moves g_k to the least significant bit, which avoids a
        // multiplication.
        const mask = 0 -% (g & 1);
        x ^= mask & v[k];
        g = g >> 1;
    }

    if (first_index < n_vectors) {
        output[first_index] = @as(f32, @floatFromInt(x)) * k_2powneg32;
    }

    // Computes the rest of the points with the stride. Generating x[i] from x[i-stride] without
    // the x in between revisits equation (**). For a stride of 16, the values from x[n] to
    // x[n+16] change the first bit 8 times, the second bit 4 times, the third bit 2 times, the
    // fourth bit once and one of the remaining bits once. Because two xors cancel out, only the
    // application of v[4], which is log2(16), and of the current direction number remain.
    const v_log2stridem1 = v[ffs(stride) - 2];
    const v_stridemask = stride - 1;

    var i = first_index + stride;
    while (i < n_vectors) : (i += stride) {
        // x[i] = x[i-stride] ^ v[b] ^ v[c], where b is log2(stride) minus 1 for the array
        // index, and c is the index of the rightmost zero bit of i without the bottom
        // log2(stride) bits, minus 1 for the array index. This is equation (**) in the paper
        // of Bratley and Fox.
        x ^= v_log2stridem1 ^ v[ffs(~((i - stride) | v_stridemask)) - 1];
        output[i] = @as(f32, @floatFromInt(x)) * k_2powneg32;
    }
}
