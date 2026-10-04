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

//! This sample is a simple code that illustrates basic usage of cooperative groups within the
//! thread block. The code launches a single thread block, creates a cooperative group of all
//! threads in the block, and a set of tiled partition cooperative groups. For each, it uses a
//! generic reduction function to calculate the sum of all the ranks in that group. In each case
//! the result is printed, together with the expected answer (which is calculated using the
//! analytical formula (n-1)*n)/2, noting that the ranks start at zero).

const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("simpleCooperativeGroups_kernel.ptx"));
    defer module.unload();
    const cgkernel = try module.function("cgkernel");

    const blocks_per_grid = 1;
    const threads_per_block = 64;

    std.log.info("Launching a single block with {d} threads...", .{threads_per_block});

    // The shared memory size is the optional launch parameter that sizes the kernel's workspace.
    try cgkernel.launch(.{
        .grid = .{ .x = blocks_per_grid },
        .block = .{ .x = threads_per_block },
        .shared_memory_bytes = threads_per_block * @sizeOf(i32),
    }, .{});
    try cuda.synchronize();

    std.log.info("...Done.", .{});
}
