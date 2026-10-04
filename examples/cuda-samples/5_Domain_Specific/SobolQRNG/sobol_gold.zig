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

//! The reference implementation of the Sobol sequence on the host.

const sobol = @import("sobol.zig");
const sobol_primitives = @import("sobol_primitives.zig").sobol_primitives;

const n_directions = sobol.n_directions;
const ffs = sobol.ffs;

const k_2powneg32: f32 = 2.3283064E-10;

/// Creates the direction numbers, based on the primitive polynomials.
pub fn initSobolDirectionVectors(n_dimensions: usize, directions: []u32) void {
    for (0..n_dimensions) |dim| {
        const v = directions[dim * n_directions ..][0..n_directions];

        if (dim == 0) {
            // The first dimension is a special case. All m's are 1.
            for (v, 0..) |*direction, i| direction.* = @as(u32, 1) << @intCast(31 - i);
        } else {
            const primitive = sobol_primitives[dim];
            const d: u5 = @intCast(primitive.degree);

            // The first direction numbers, up to the degree of the polynomial, are simply
            // v[i] = m[i] / 2^i, stored in Q0.32 format.
            for (0..d) |i| {
                v[i] = primitive.m[i] << @intCast(31 - i);
            }

            // The remaining direction numbers are computed as described in the paper of Bratley
            // and Fox:
            // v[i] = a[1]v[i-1] ^ a[2]v[i-2] ^ ... ^ a[v-1]v[i-d+1] ^ v[i-d] ^ v[i-d]/2^d
            for (d..n_directions) |i| {
                // First the v[i-d] ^ v[i-d]/2^d part.
                v[i] = v[i - d] ^ (v[i - d] >> d);

                // Then the a[1]v[i-1] ^ a[2]v[i-2] ^ ... part. The coefficients a[] are zero or
                // one. The input tables store them as bits of a single integer, so the code
                // extracts a coefficient with a right shift and a mask with 1. For example, a
                // polynomial of degree 10 has ten useful bits in a. To get a[2], the code shifts
                // 7 times to move the 8th bit to the least significant bit, and masks with 1.
                for (1..d) |j| {
                    v[i] ^= ((primitive.a >> @intCast(d - 1 - j)) & 1) * v[i - j];
                }
            }
        }
    }
}

/// The reference model for generating Sobol numbers on the host.
pub fn sobolCpu(
    n_vectors: usize,
    n_dimensions: usize,
    directions: []const u32,
    output: []f32,
) void {
    for (0..n_dimensions) |d| {
        const v = directions[d * n_directions ..][0..n_directions];
        var x: u32 = 0;
        // x[0] is zero in all dimensions.
        output[n_vectors * d] = 0.0;

        for (1..n_vectors) |i| {
            // x[i] = x[i-1] ^ v[c], where c is the index of the rightmost zero bit in i, minus 1
            // because array indices count from zero. This is equation (**) in the paper of
            // Bratley and Fox.
            x ^= v[ffs(~@as(u32, @intCast(i - 1))) - 1];
            output[i + n_vectors * d] = @as(f32, @floatFromInt(x)) * k_2powneg32;
        }
    }
}
