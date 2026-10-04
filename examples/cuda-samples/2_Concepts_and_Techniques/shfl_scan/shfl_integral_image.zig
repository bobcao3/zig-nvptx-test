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

const Uint4 = @Vector(4, u32);
const Uchar4 = @Vector(4, u8);

inline fn uintToUchar4(in: u32) Uchar4 {
    return @bitCast(in);
}

const PackedResult = struct {
    x: Uint4,
    y: Uint4,
    z: Uint4,
    w: Uint4,
};

var prefix_sums: [128]u32 addrspace(.shared) = undefined;

inline fn shflUp(value: u32, delta: u32) u32 {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.up, std.nvptx.warp_size);
    return std.nvptx.shflSync(.up, u32, value, delta, clamp, .all);
}

inline fn shflXor(value: Uint4, lane_mask: u32) Uint4 {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.bfly, std.nvptx.warp_size);
    var result: Uint4 = undefined;
    inline for (0..4) |i| {
        result[i] = std.nvptx.shflSync(.bfly, u32, value[i], lane_mask, clamp, .all);
    }
    return result;
}

inline fn getPrefixSum(data: Uint4) PackedResult {
    const lane_id = @workItemId(0) % std.nvptx.warp_size;
    const warp_id = @workItemId(0) / std.nvptx.warp_size;

    var result: [16]u32 = @splat(0);
    inline for (0..4) |word| {
        const bytes: Uint4 = @intCast(uintToUchar4(data[word]));

        result[4 * word + 0] = bytes[0];
        result[4 * word + 1] = bytes[0] + bytes[1];
        result[4 * word + 2] = bytes[0] + bytes[1] + bytes[2];
        result[4 * word + 3] = bytes[0] + bytes[1] + bytes[2] + bytes[3];
    }

    inline for (4..8) |i| result[i] += result[3];
    inline for (8..12) |i| result[i] += result[7];
    inline for (12..16) |i| result[i] += result[11];

    var sum = result[15];

    // The prefix sum of the 16 values of each work item is complete. A shuffle scan distributes
    // the sums (`result[15]`) to the work items that follow in the warp.
    inline for (.{ 1, 2, 4, 8, 16 }) |i| {
        const n = shflUp(sum, i);

        if (lane_id >= i) {
            inline for (0..16) |j| result[j] += n;

            sum += n;
        }
    }

    // The last work item of each warp stores the sum of the warp in shared memory. The first warp
    // then scans these sums with shuffles, and the results are added to the other warps.
    if (lane_id == std.nvptx.warp_size - 1) prefix_sums[warp_id] = result[15];

    std.nvptx.barrierSyncAligned(.work_group, null);

    if (warp_id == 0) {
        var warp_sum = prefix_sums[lane_id];

        inline for (.{ 1, 2, 4, 8, 16 }) |i| {
            const n = shflUp(warp_sum, i);

            if (lane_id >= i) warp_sum += n;
        }

        prefix_sums[lane_id] = warp_sum;
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    if (warp_id > 0) {
        const block_sum = prefix_sums[warp_id - 1];

        inline for (0..16) |i| result[i] += block_sum;
    }

    return .{
        .x = result[0..4].*,
        .y = result[4..8].*,
        .z = result[8..12].*,
        .w = result[12..16].*,
    };
}

inline fn idxToElem(result: PackedResult, idx: u32) Uint4 {
    return switch (idx) {
        0 => result.x,
        1 => result.y,
        2 => result.z,
        3 => result.w,
        else => @splat(0),
    };
}

/// This function demonstrates some uses of the shuffle instruction in the generation of an
/// integral image (also called a summed area table). The approach is two pass, a horizontal
/// (scanline) then a vertical (column) pass. This is the horizontal pass kernel.
pub fn shflIntimageRows(
    img: [*]addrspace(.global) const Uint4,
    integral_image: [*]addrspace(.global) Uint4,
) callconv(.kernel) void {
    const id = @workItemId(0);
    const scanline = img + @workGroupId(0) * 120;
    var result = getPrefixSum(scanline[id]);

    // Each work item has 16 values to write, which are now integer data (to avoid overflow).
    // Instead of each work item writing consecutive `Uint4`s, the approach shown here uses the
    // shuffle command to reformat the data inside the registers so that each work item holds
    // consecutive data to be written, so that larger contiguous segments can be assembled for
    // writing.
    //
    // For example, data that needs to be written as
    //
    //   GMEM[16] <- x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3 w0 w1 w2 w3
    //
    // but is stored in registers (r0..r3), in four work items (0..3) as:
    //
    //   work item  0  1  2  3
    //     r0      x0 y0 z0 w0
    //     r1      x1 y1 z1 w1
    //     r2      x2 y2 z2 w2
    //     r3      x3 y3 z3 w3
    //
    // after applying xor shuffles to move data between registers r1..r3:
    //
    //   work item 00 01 10 11
    //             x0 y0 z0 w0
    //    xor(01)->y1 x1 w1 z1
    //    xor(10)->z2 w2 x2 y2
    //    xor(11)->w3 z3 y3 x3
    //
    // and now x0..x3 and z0..z3 can be written out in order by all work items.
    //
    // In the code below, each register above holds four integers that are written as one `Uint4`
    // to global memory.
    const id_mask = id & 3;
    const id_swizzle = (id + 2) & 3;
    const id_shift = (id >> 2) << 4;
    const block_offset = @workGroupId(0) * 480;

    result.y = shflXor(result.y, 1);
    result.z = shflXor(result.z, 2);
    result.w = shflXor(result.w, 3);

    integral_image[block_offset + id_mask + id_shift] = idxToElem(result, id_mask);
    integral_image[block_offset + id_swizzle + id_shift + 8] = idxToElem(result, id_swizzle);

    // The xor shuffle below places the y0..y3 and w0..w3 data in order.
    result.x = shflXor(result.x, 1);
    result.y = shflXor(result.y, 1);
    result.z = shflXor(result.z, 1);
    result.w = shflXor(result.w, 1);

    integral_image[block_offset + id_mask + id_shift + 4] = idxToElem(result, id_mask);
    integral_image[block_offset + id_swizzle + id_shift + 12] = idxToElem(result, id_swizzle);
}

var column_sums: [32][9]u32 addrspace(.shared) = undefined;

/// This kernel computes columnwise prefix sums. When the data input is the row sums from above,
/// this completes the integral image. The approach here is to have each work group compute a
/// local set of sums. First, the data covered by the work group is loaded into shared memory.
/// Then, instead of performing a sum in shared memory with a barrier between stages, the data is
/// reformatted so that the necessary sums occur inside warps and the shuffle scan operation is
/// used. The final set of sums from the work group is then propagated, with the work group
/// computing "down" the image and adding the running sum to the local sums.
pub fn shflVerticalShfl(
    img: [*]addrspace(.global) u32,
    width: u32,
    height: u32,
) callconv(.kernel) void {
    _ = height;
    const x = @workItemId(0);
    const y = @workItemId(1);
    const tidx = @workGroupId(0) * @workGroupSize(0) + x;
    const lane_id = tidx % 8;
    var step_sum: u32 = 0;

    column_sums[x][y] = 0;
    std.nvptx.barrierSyncAligned(.work_group, null);

    for (0..135) |step| {
        var sum: u32 = 0;
        const p = &img[(y + step * 8) * width + tidx];

        sum = p.*;
        column_sums[x][y] = sum;
        std.nvptx.barrierSyncAligned(.work_group, null);

        // Place the values into shared memory, then shuffle scan them, reformatting so that the
        // column sums are computed within a warp, and read the results out properly.
        const j = x % 8;
        const k = x / 8 + y * 4;

        var partial_sum = column_sums[k][j];

        var i: u32 = 1;
        while (i <= 8) : (i *= 2) {
            const n = std.nvptx.shflSync(.up, u32, partial_sum, i, .fromWidth(.up, 32), .all);

            if (lane_id >= i) partial_sum += n;
        }

        column_sums[k][j] = partial_sum;
        std.nvptx.barrierSyncAligned(.work_group, null);

        if (y > 0) sum += column_sums[x][y - 1];

        sum += step_sum;
        step_sum += column_sums[x][@workGroupSize(1) - 1];
        std.nvptx.barrierSyncAligned(.work_group, null);
        p.* = sum;
    }
}
