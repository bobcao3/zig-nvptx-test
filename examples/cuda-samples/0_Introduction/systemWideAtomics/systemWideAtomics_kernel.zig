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

//! A program demonstrating trivial use of system-wide atomics on migratable memory.

const std = @import("std");

const loop_num = 50;

export fn atomicKernel(atom_arr: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = @workGroupSize(0) * @workGroupId(0) + @workItemId(0);
    const signed_tid: i32 = @bitCast(tid);
    const unsigned_arr: [*]addrspace(.global) u32 = @ptrCast(atom_arr);

    for (0..loop_num) |_| {
        _ = std.nvptx.atom(.sys, .monotonic, .add, &atom_arr[0], 10);

        _ = std.nvptx.atom(.sys, .monotonic, .exch, &atom_arr[1], signed_tid);

        _ = std.nvptx.atom(.sys, .monotonic, .max, &atom_arr[2], signed_tid);

        _ = std.nvptx.atom(.sys, .monotonic, .min, &atom_arr[3], signed_tid);

        _ = std.nvptx.atom(.sys, .monotonic, .inc, &unsigned_arr[4], 17);

        _ = std.nvptx.atom(.sys, .monotonic, .dec, &unsigned_arr[5], 137);

        _ = std.nvptx.atomCas(.sys, .monotonic, &atom_arr[6], signed_tid -% 1, signed_tid);

        _ = std.nvptx.atom(.sys, .monotonic, .@"and", &atom_arr[7], @bitCast(2 *% tid +% 7));

        _ = std.nvptx.atom(.sys, .monotonic, .@"or", &atom_arr[8], std.math.shl(i32, 1, tid));

        _ = std.nvptx.atom(.sys, .monotonic, .xor, &atom_arr[9], signed_tid);
    }
}
