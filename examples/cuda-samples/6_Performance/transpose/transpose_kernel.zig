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

//! Transpose kernels for a floating-point matrix. The kernels improve performance step by step
//! through coalescing and the removal of shared memory bank conflicts. Several kernels perform a
//! copy, which represents the best case performance that a transpose can achieve. Each work group
//! transposes or copies a tile of `tile_dim` x `tile_dim` elements with `tile_dim` x `block_rows`
//! work items, so that each work item handles `tile_dim / block_rows` elements.

const std = @import("std");
const common = @import("transpose_common.zig");

const tile_dim = common.tile_dim;
const block_rows = common.block_rows;
const passes = tile_dim / block_rows;

const Matrix = [*]addrspace(.global) f32;
const ConstMatrix = [*]addrspace(.global) const f32;

var tile: [tile_dim][tile_dim]f32 addrspace(.shared) = undefined;
var padded_tile: [tile_dim][tile_dim + 1]f32 addrspace(.shared) = undefined;

inline fn thread(comptime dimension: u32) u32 {
    return @workItemId(dimension);
}

inline fn block(comptime dimension: u32) u32 {
    return @workGroupId(dimension);
}

/// Copies. The width and the height must be integral multiples of `tile_dim`.
export fn copy(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    _ = height;
    const x_index = block(0) * tile_dim + thread(0);
    const y_index = block(1) * tile_dim + thread(1);

    const index = x_index + width * y_index;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index + i * width] = idata[index + i * width];
    }
}

export fn copySharedMem(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    const x_index = block(0) * tile_dim + thread(0);
    const y_index = block(1) * tile_dim + thread(1);

    const index = x_index + width * y_index;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        if (x_index < width and y_index < height) {
            tile[thread(1) + i][thread(0)] = idata[index + i * width];
            tile[thread(1) + i][thread(0)] = idata[index + i * width];
        }
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        if (x_index < height and y_index < width) {
            odata[index + i * width] = tile[thread(1) + i][thread(0)];
            odata[index + i * width] = tile[thread(1) + i][thread(0)];
        }
    }
}

/// Transposes. The width and the height must be integral multiples of `tile_dim`.
export fn transposeNaive(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    const x_index = block(0) * tile_dim + thread(0);
    const y_index = block(1) * tile_dim + thread(1);

    const index_in = x_index + width * y_index;
    const index_out = y_index + height * x_index;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index_out + i] = idata[index_in + i * width];
    }
}

/// Coalesced transpose, with bank conflicts.
export fn transposeCoalesced(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    var x_index = block(0) * tile_dim + thread(0);
    var y_index = block(1) * tile_dim + thread(1);
    const index_in = x_index + y_index * width;

    x_index = block(1) * tile_dim + thread(0);
    y_index = block(0) * tile_dim + thread(1);
    const index_out = x_index + y_index * height;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        tile[thread(1) + i][thread(0)] = idata[index_in + i * width];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index_out + i * height] = tile[thread(0)][thread(1) + i];
    }
}

/// Coalesced transpose without bank conflicts.
export fn transposeNoBankConflicts(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    var x_index = block(0) * tile_dim + thread(0);
    var y_index = block(1) * tile_dim + thread(1);
    const index_in = x_index + y_index * width;

    x_index = block(1) * tile_dim + thread(0);
    y_index = block(0) * tile_dim + thread(1);
    const index_out = x_index + y_index * height;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        padded_tile[thread(1) + i][thread(0)] = idata[index_in + i * width];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index_out + i * height] = padded_tile[thread(0)][thread(1) + i];
    }
}

/// Transpose that reorders the execution of work groups along the diagonals of the matrix. It is
/// also coalesced and has no bank conflicts. The first work group index is the distance along a
/// diagonal and the second work group index selects the diagonal. `block_x` and `block_y` map the
/// diagonal coordinates to Cartesian coordinates.
export fn transposeDiagonal(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    const grid_x = std.nvptx.nctaid(0);
    const grid_y = std.nvptx.nctaid(1);

    const block_x: u32, const block_y: u32 = if (width == height) .{
        (block(0) + block(1)) % grid_x,
        block(0),
    } else diagonal: {
        const bid = block(0) + grid_x * block(1);
        const y = bid % grid_y;
        break :diagonal .{ ((bid / grid_y) + y) % grid_x, y };
    };

    var x_index = block_x * tile_dim + thread(0);
    var y_index = block_y * tile_dim + thread(1);
    const index_in = x_index + y_index * width;

    x_index = block_y * tile_dim + thread(0);
    y_index = block_x * tile_dim + thread(1);
    const index_out = x_index + y_index * height;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        padded_tile[thread(1) + i][thread(0)] = idata[index_in + i * width];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index_out + i * height] = padded_tile[thread(0)][thread(1) + i];
    }
}

/// Partial transposes. The fine-grained and the coarse-grained kernels perform only part of a
/// transpose and fail a comparison with the reference solution. They assess the performance of
/// the components of a full transpose.
export fn transposeFineGrained(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    const x_index = block(0) * tile_dim + thread(0);
    const y_index = block(1) * tile_dim + thread(1);
    const index = x_index + y_index * width;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        padded_tile[thread(1) + i][thread(0)] = idata[index + i * width];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index + i * height] = padded_tile[thread(0)][thread(1) + i];
    }
}

export fn transposeCoarseGrained(
    odata: Matrix,
    idata: ConstMatrix,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    var x_index = block(0) * tile_dim + thread(0);
    var y_index = block(1) * tile_dim + thread(1);
    const index_in = x_index + y_index * width;

    x_index = block(1) * tile_dim + thread(0);
    y_index = block(0) * tile_dim + thread(1);
    const index_out = x_index + y_index * height;

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        padded_tile[thread(1) + i][thread(0)] = idata[index_in + i * width];
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (0..passes) |pass| {
        const i = pass * block_rows;
        odata[index_out + i * height] = padded_tile[thread(1) + i][thread(0)];
    }
}
