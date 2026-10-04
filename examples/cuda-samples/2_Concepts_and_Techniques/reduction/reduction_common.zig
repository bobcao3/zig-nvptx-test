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

/// The numbers of work items per work group that the kernels 4 to 6 are compiled for.
pub const block_sizes = [_]u32{ 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };

pub const block_sizes_7 = [_]u32{1024} ++ block_sizes;

pub const multi_warp_block_sizes = [_]u32{ 1024, 512, 256, 128, 64 };

pub const multi_warp_groups = 2;

pub fn kernelName(
    comptime kernel: u32,
    comptime T: type,
    comptime block_size: ?u32,
    comptime n_is_pow2: ?bool,
) [:0]const u8 {
    @setEvalBranchQuota(100_000);
    const size = if (block_size) |size| std.fmt.comptimePrint("_{d}", .{size}) else "";
    const pow2 = if (n_is_pow2) |pow2| (if (pow2) "_pow2" else "_any") else "";
    return std.fmt.comptimePrint("reduce{d}_{s}{s}{s}", .{ kernel, @typeName(T), size, pow2 });
}
