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

/// Kernel #1 tests the across-the-warp vote(any) intrinsic. If ANY one of the work items (within
/// the warp) of the predicated condition returns a non-zero value, then all work items within
/// this warp will return a non-zero value.
export fn voteAnyKernel1(
    input: [*]addrspace(.global) const u32,
    result: [*]addrspace(.global) u32,
    size: i32,
) callconv(.kernel) void {
    _ = size;
    const tx = @workItemId(0);

    result[tx] = @intFromBool(std.nvptx.voteSync(.any, input[tx] != 0, .all));
}

/// Kernel #2 tests the across-the-warp vote(all) intrinsic. If ALL of the work items (within the
/// warp) of the predicated condition returns a non-zero value, then all work items within this
/// warp will return a non-zero value.
export fn voteAllKernel2(
    input: [*]addrspace(.global) const u32,
    result: [*]addrspace(.global) u32,
    size: i32,
) callconv(.kernel) void {
    _ = size;
    const tx = @workItemId(0);

    result[tx] = @intFromBool(std.nvptx.voteSync(.all, input[tx] != 0, .all));
}

/// Kernel #3 is a directed test for the across-the-warp vote(all) intrinsic. This kernel will
/// test for conditions across warps, and within half warps.
export fn voteAnyKernel3(info: [*]addrspace(.global) bool, warp_size: i32) callconv(.kernel) void {
    const tx = @workItemId(0);
    const offs = info + tx * 3;

    // The following should hold true for the second and third warp.
    offs[0] = std.nvptx.voteSync(.any, tx >= @divTrunc(warp_size * 3, 2), .all);
    // The following should hold true for the "upper half" of the second warp, and all of the
    // third warp.
    offs[1] = tx >= @divTrunc(warp_size * 3, 2);

    // The following should hold true for the third warp only.
    if (std.nvptx.voteSync(.all, tx >= @divTrunc(warp_size * 3, 2), .all)) {
        offs[2] = true;
    }
}
