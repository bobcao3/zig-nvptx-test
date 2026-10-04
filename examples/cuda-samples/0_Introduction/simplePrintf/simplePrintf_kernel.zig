// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/// Each work item calls device-side printf, printing its linear block index, its linear thread
/// index, and `val`. Output is flushed on host sync; line order is not deterministic since blocks
/// and work items run concurrently.
export fn simplePrintfKernel(val: i32) callconv(.kernel) void {
    const Arguments = extern struct { block: i32, thread: i32, val: i32 };
    const block_index = @workGroupId(1) * std.nvptx.nctaid(0) + @workGroupId(0);
    const thread_index = @workItemId(2) * @workGroupSize(0) * @workGroupSize(1) +
        @workItemId(1) * @workGroupSize(0) + @workItemId(0);
    const arguments: Arguments = .{
        .block = @intCast(block_index),
        .thread = @intCast(thread_index),
        .val = val,
    };
    _ = std.nvptx.vprintf("[%d, %d]:\t\tValue is:%d\n", &arguments);
}
