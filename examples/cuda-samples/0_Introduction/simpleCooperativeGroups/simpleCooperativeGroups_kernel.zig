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

const ThreadBlock = struct {
    inline fn size(_: ThreadBlock) u32 {
        return @workGroupSize(0) * @workGroupSize(1) * @workGroupSize(2);
    }

    inline fn threadRank(_: ThreadBlock) u32 {
        return (@workItemId(2) * @workGroupSize(1) + @workItemId(1)) * @workGroupSize(0) +
            @workItemId(0);
    }

    inline fn sync(_: ThreadBlock) void {
        std.nvptx.barrierSyncAligned(.work_group, null);
    }
};

/// A group of `tile_size` consecutive work items of a warp, as `tiled_partition<tile_size>`.
fn ThreadBlockTile(comptime tile_size: u32) type {
    return struct {
        block: ThreadBlock,

        inline fn size(_: @This()) u32 {
            return tile_size;
        }

        inline fn threadRank(tile: @This()) u32 {
            return tile.block.threadRank() % tile_size;
        }

        inline fn sync(_: @This()) void {
            const mask = (@as(u32, 1) << tile_size) - 1;
            std.nvptx.barWarpSync(@fromBackingInt(mask << @intCast(std.nvptx.laneid() / tile_size * tile_size)));
        }
    };
}

/// Calculates the sum of `val` across the group `g`. The workspace array `x` must be large enough
/// to contain `g.size()` integers.
inline fn sumReduction(g: anytype, x: [*]addrspace(.shared) i32, val_in: i32) i32 {
    var val = val_in;
    const lane = g.threadRank();

    // In each iteration the number of active work items, `i`, is halved, and each active work item
    // adds the value of its partner `lane + i`.
    var i = g.size() / 2;
    while (i > 0) : (i /= 2) {
        x[lane] = val;
        g.sync();
        if (lane < i) val += x[lane + i];
        g.sync();
    }

    return if (g.threadRank() == 0) val else -1;
}

extern var workspace: [0]i32 align(16) addrspace(.shared);

export fn cgkernel() callconv(.kernel) void {
    const thread_block_group: ThreadBlock = .{};
    const thread_block_group_size: i32 = @intCast(thread_block_group.size());
    const shared_workspace: [*]addrspace(.shared) i32 = &workspace;

    var input: i32 = @intCast(thread_block_group.threadRank());
    var expected_output = @divExact((thread_block_group_size - 1) * thread_block_group_size, 2);

    var output = sumReduction(thread_block_group, shared_workspace, input);

    if (thread_block_group.threadRank() == 0) {
        const sum_arguments: extern struct { last_rank: i32, output: i32, expected: i32 } = .{
            .last_rank = thread_block_group_size - 1,
            .output = output,
            .expected = expected_output,
        };
        _ = std.nvptx.vprintf(" Sum of all ranks 0..%d in threadBlockGroup is %d (expected %d)\n\n", &sum_arguments);
        const groups_arguments: extern struct { groups: i32 } = .{
            .groups = @divTrunc(thread_block_group_size, 16),
        };
        _ = std.nvptx.vprintf(" Now creating %d groups, each of size 16 threads:\n\n", &groups_arguments);
    }

    thread_block_group.sync();

    const tiled_partition16: ThreadBlockTile(16) = .{ .block = thread_block_group };

    // Gives each group its own area of the workspace.
    const workspace_offset = thread_block_group.threadRank() - tiled_partition16.threadRank();

    input = @intCast(tiled_partition16.threadRank());
    expected_output = 15 * 16 / 2;

    output = sumReduction(tiled_partition16, shared_workspace + workspace_offset, input);

    if (tiled_partition16.threadRank() == 0) {
        const arguments: extern struct { output: i32, expected: i32 } = .{
            .output = output,
            .expected = expected_output,
        };
        _ = std.nvptx.vprintf("   Sum of all ranks 0..15 in this tiledPartition16 group is %d " ++
            "(expected %d)\n", &arguments);
    }
}
