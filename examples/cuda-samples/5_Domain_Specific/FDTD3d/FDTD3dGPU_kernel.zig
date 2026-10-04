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

//! The finite differences kernel of the FDTD3d sample: a 3D stencil of radius 4 that sweeps
//! through the volume one xy-plane at a time.

const std = @import("std");
const constants = @import("constants.zig");

const radius = constants.radius;

export var stencil: [radius + 1]f32 addrspace(.global) = undefined;

var tile: [constants.block_dim_max_y + 2 * radius][constants.block_dim_x + 2 * radius]f32 addrspace(.shared) = undefined;

export fn finiteDifferencesKernel(
    output: [*]addrspace(.global) f32,
    input: [*]addrspace(.global) const f32,
    dimx: i32,
    dimy: i32,
    dimz: i32,
) callconv(.kernel) void {
    var validr = true;
    var validw = true;
    const gtidx: i32 = @intCast(@workGroupId(0) * @workGroupSize(0) + @workItemId(0));
    const gtidy: i32 = @intCast(@workGroupId(1) * @workGroupSize(1) + @workItemId(1));
    const ltidx: i32 = @intCast(@workItemId(0));
    const ltidy: i32 = @intCast(@workItemId(1));
    const workx: i32 = @intCast(@workGroupSize(0));
    const worky: i32 = @intCast(@workGroupSize(1));

    const stride_y = dimx + 2 * radius;
    const stride_z = stride_y * (dimy + 2 * radius);

    var input_index: i32 = 0;
    var output_index: i32 = 0;

    // Advance `input_index` to the start of the inner volume.
    input_index += radius * stride_y + radius;

    // Advance `input_index` to the target element.
    input_index += gtidy * stride_y + gtidx;

    var infront: [radius]f32 = undefined;
    var behind: [radius]f32 = undefined;
    var current: f32 = undefined;

    const tx: usize = @intCast(ltidx + radius);
    const ty: usize = @intCast(ltidy + radius);

    // Check in bounds.
    if (gtidx >= dimx + radius or gtidy >= dimy + radius) validr = false;

    if (gtidx >= dimx or gtidy >= dimy) validw = false;

    // Preload the "infront" and "behind" data.
    inline for (0..radius - 1) |step| {
        const i = radius - 2 - step;
        if (validr) behind[i] = input[@intCast(input_index)];

        input_index += stride_z;
    }

    if (validr) current = input[@intCast(input_index)];

    output_index = input_index;
    input_index += stride_z;

    inline for (0..radius) |i| {
        if (validr) infront[i] = input[@intCast(input_index)];

        input_index += stride_z;
    }

    // Step through the xy-planes.
    var iz: i32 = 0;
    while (iz < dimz) : (iz += 1) {
        // Advance the slice (move the thread-front).
        inline for (0..radius - 1) |step| {
            const i = radius - 1 - step;
            behind[i] = behind[i - 1];
        }

        behind[0] = current;
        current = infront[0];

        inline for (0..radius - 1) |i| infront[i] = infront[i + 1];

        if (validr) infront[radius - 1] = input[@intCast(input_index)];

        input_index += stride_z;
        output_index += stride_z;
        std.nvptx.barrierSyncAligned(.work_group, null);

        // For the work items on the boundary of the problem, the index that reads the halo may
        // wrap to the previous or next row or even the previous or next xy-plane. This is
        // acceptable since the output write is disabled for these work items and at least one
        // xy-plane exists before and after the current plane, so the access stays within bounds.

        // Update the data slice in the local tile. First the halo above and below.
        if (ltidy < radius) {
            const row: usize = @intCast(ltidy);
            tile[row][tx] = input[@intCast(output_index - radius * stride_y)];
            tile[@intCast(ltidy + worky + radius)][tx] = input[@intCast(output_index + worky * stride_y)];
        }

        // The halo left and right.
        if (ltidx < radius) {
            const column: usize = @intCast(ltidx);
            tile[ty][column] = input[@intCast(output_index - radius)];
            tile[ty][@intCast(ltidx + workx + radius)] = input[@intCast(output_index + workx)];
        }

        tile[ty][tx] = current;
        std.nvptx.barrierSyncAligned(.work_group, null);

        // Compute the output value.
        var value = stencil[0] * current;

        inline for (1..radius + 1) |i| {
            value += stencil[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value.
        if (validw) output[@intCast(output_index)] = value;
    }
}
