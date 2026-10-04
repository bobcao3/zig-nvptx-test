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

//! The permutations of the cluster assignment of the 16 colors of a block.

const std = @import("std");

pub fn computePermutations(permutations: *[1024]u32) void {
    var indices: [16]u32 = undefined;
    var num: usize = 0;

    // 3 element permutations:

    // The first cluster [0, i) is at the start.
    @memset(&indices, 0);

    const imax = 15;

    var i: i32 = imax;
    while (i >= 0) : (i -= 1) {
        // The second cluster [i, j) is half along.
        for (@intCast(i)..16) |m| indices[m] = 2;

        const jmax: i32 = if (i == 0) 15 else 16;

        var j = jmax;
        while (j >= i) : (j -= 1) {
            // The last cluster [j, k) is at the end.
            if (j < 16) {
                indices[@intCast(j)] = 1;
            }

            var permutation: u32 = 0;

            for (0..16) |p| {
                permutation |= indices[p] << @intCast(p * 2);
            }

            permutations[num] = permutation;

            num += 1;
        }
    }

    std.debug.assert(num == 151);

    for (0..9) |_| {
        permutations[num] = 0x000AA555;
        num += 1;
    }

    std.debug.assert(num == 160);

    // Append 4 element permutations:

    // The first cluster [0, i) is at the start.
    @memset(&indices, 0);

    i = imax;
    while (i >= 0) : (i -= 1) {
        // The second cluster [i, j) is one third along.
        for (@intCast(i)..16) |m| indices[m] = 2;

        const jmax: i32 = if (i == 0) 15 else 16;

        var j = jmax;
        while (j >= i) : (j -= 1) {
            // The third cluster [j, k) is two thirds along.
            for (@intCast(j)..16) |m| indices[m] = 3;

            const kmax: i32 = if (j == 0) 15 else 16;

            var k = kmax;
            while (k >= j) : (k -= 1) {
                // The last cluster [k, n) is at the end.
                if (k < 16) {
                    indices[@intCast(k)] = 1;
                }

                var permutation: u32 = 0;

                var has_three = false;

                for (0..16) |p| {
                    permutation |= indices[p] << @intCast(p * 2);

                    if (indices[p] == 3) has_three = true;
                }

                if (has_three) {
                    permutations[num] = permutation;
                    num += 1;
                }
            }
        }
    }

    std.debug.assert(num == 975);

    // The remaining 49 elements could hold better values.
    for (0..49) |_| {
        permutations[num] = 0x00AAFF55;
        num += 1;
    }

    std.debug.assert(num == 1024);
}
