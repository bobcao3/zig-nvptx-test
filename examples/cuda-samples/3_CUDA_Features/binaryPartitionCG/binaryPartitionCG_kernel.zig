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

//! This sample illustrates basic usage of binary partition cooperative groups within the thread
//! block tile when divergent path exists.
//! 1. Each thread loads a value from random array.
//! 2. Then it checks if the value is odd or even.
//! 3. It creates a binary partition group based on the above predicate.
//! 4. It counts the number of odd and even values in the group based on the size of the binary
//!    groups.
//! 5. It writes the count to a global counter of odd values.
//! 6. It sums the values loaded by the individual threads (using reduce) and writes the sums to
//!    the global even and odd element sums.
//!
//! **NOTE**:
//! binary_partition results in splitting warp into divergent thread groups. This is not good from
//! performance perspective, but in cases where warp divergence is inevitable one can use
//! binary_partition group.

const std = @import("std");

/// Creates cooperative groups and performs odd/even counting and summation.
export fn oddEvenCountAndSumCg(
    input_arr: [*]addrspace(.global) const i32,
    num_of_odds: *addrspace(.global) i32,
    sum_of_odd_and_evens: *addrspace(.global) [2]i32,
    size: u32,
) callconv(.kernel) void {
    const grid_size = std.nvptx.nctaid(0) * @workGroupSize(0);
    var i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    while (i < size) : (i += grid_size) {
        const elem = input_arr[i];
        const is_odd = elem & 1 != 0;
        const ballot = std.nvptx.voteBallotSync(is_odd, .all);
        const sub_tile_mask = if (is_odd) ballot else ~ballot;
        const sub_tile_size: i32 = @popCount(sub_tile_mask);
        const sub_tile_rank = @popCount(sub_tile_mask & std.nvptx.lanemaskLt());
        const group_sum = std.nvptx.reduxSync(.add, i32, elem, @fromBackingInt(sub_tile_mask));
        if (is_odd) {
            if (sub_tile_rank == 0) {
                _ = @atomicRmw(i32, num_of_odds, .Add, sub_tile_size, .monotonic);
                _ = @atomicRmw(i32, &sum_of_odd_and_evens[0], .Add, group_sum, .monotonic);
            }
        } else {
            if (sub_tile_rank == 0) {
                _ = @atomicRmw(i32, &sum_of_odd_and_evens[1], .Add, group_sum, .monotonic);
            }
        }
        std.nvptx.barWarpSync(.all);
    }
}
