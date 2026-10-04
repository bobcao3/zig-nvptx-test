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

//! Separable convolution of a 2D image with a filter kernel. The row pass and the column pass
//! each load a tile with halos into shared memory.

const std = @import("std");
const common = @import("convolutionSeparable_common.zig");

const kernel_radius = common.kernel_radius;

export var c_kernel: [common.kernel_length]f32 addrspace(.global) = undefined;

const rows = struct {
    const blockdim_x = common.rows_blockdim_x;
    const blockdim_y = common.rows_blockdim_y;
    const result_steps = common.rows_result_steps;
    const halo_steps = common.rows_halo_steps;

    var s_data: [blockdim_y][(result_steps + 2 * halo_steps) * blockdim_x]f32 addrspace(.shared) =
        undefined;
};

export fn convolutionRowsKernel(
    d_dst: [*]addrspace(.global) f32,
    d_src: [*]addrspace(.global) const f32,
    image_w: i32,
    image_h: i32,
    pitch: i32,
) callconv(.kernel) void {
    _ = image_h;
    const blockdim_x = rows.blockdim_x;
    const blockdim_y = rows.blockdim_y;
    const result_steps = rows.result_steps;
    const halo_steps = rows.halo_steps;
    const s_data = &rows.s_data;
    const thread_x: i32 = @intCast(@workItemId(0));
    const thread_y: i32 = @intCast(@workItemId(1));

    const base_x = (@as(i32, @intCast(@workGroupId(0))) * result_steps - halo_steps) * blockdim_x +
        thread_x;
    const base_y = @as(i32, @intCast(@workGroupId(1))) * blockdim_y + thread_y;
    const offset = base_y * pitch + base_x;
    const row: usize = @intCast(thread_y);
    const column: usize = @intCast(thread_x);

    inline for (halo_steps..halo_steps + result_steps) |i| {
        const shift: i32 = i * blockdim_x;
        s_data[row][column + i * blockdim_x] = d_src[@intCast(offset + shift)];
    }

    inline for (0..halo_steps) |i| {
        const shift: i32 = i * blockdim_x;
        s_data[row][column + i * blockdim_x] =
            if (base_x >= -shift) d_src[@intCast(offset + shift)] else 0;
    }

    inline for (halo_steps + result_steps..halo_steps + result_steps + halo_steps) |i| {
        const shift: i32 = i * blockdim_x;
        s_data[row][column + i * blockdim_x] =
            if (image_w - base_x > shift) d_src[@intCast(offset + shift)] else 0;
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (halo_steps..halo_steps + result_steps) |i| {
        var sum: f32 = 0;
        inline for (0..2 * kernel_radius + 1) |j| {
            const x = column + i * blockdim_x + j - kernel_radius;
            sum += c_kernel[2 * kernel_radius - j] * s_data[row][x];
        }
        const shift: i32 = i * blockdim_x;
        d_dst[@intCast(offset + shift)] = sum;
    }
}

const columns = struct {
    const blockdim_x = common.columns_blockdim_x;
    const blockdim_y = common.columns_blockdim_y;
    const result_steps = common.columns_result_steps;
    const halo_steps = common.columns_halo_steps;

    const row_length = (result_steps + 2 * halo_steps) * blockdim_y + 1;
    var s_data: [blockdim_x][row_length]f32 addrspace(.shared) = undefined;
};

export fn convolutionColumnsKernel(
    d_dst: [*]addrspace(.global) f32,
    d_src: [*]addrspace(.global) const f32,
    image_w: i32,
    image_h: i32,
    pitch: i32,
) callconv(.kernel) void {
    _ = image_w;
    const blockdim_x = columns.blockdim_x;
    const blockdim_y = columns.blockdim_y;
    const result_steps = columns.result_steps;
    const halo_steps = columns.halo_steps;
    const s_data = &columns.s_data;
    const thread_x: i32 = @intCast(@workItemId(0));
    const thread_y: i32 = @intCast(@workItemId(1));

    const base_x = @as(i32, @intCast(@workGroupId(0))) * blockdim_x + thread_x;
    const base_y = (@as(i32, @intCast(@workGroupId(1))) * result_steps - halo_steps) * blockdim_y +
        thread_y;
    const offset = base_y * pitch + base_x;
    const row: usize = @intCast(thread_x);
    const column: usize = @intCast(thread_y);

    inline for (halo_steps..halo_steps + result_steps) |i| {
        const shift: i32 = i * blockdim_y;
        s_data[row][column + i * blockdim_y] = d_src[@intCast(offset + shift * pitch)];
    }

    inline for (0..halo_steps) |i| {
        const shift: i32 = i * blockdim_y;
        s_data[row][column + i * blockdim_y] =
            if (base_y >= -shift) d_src[@intCast(offset + shift * pitch)] else 0;
    }

    inline for (halo_steps + result_steps..halo_steps + result_steps + halo_steps) |i| {
        const shift: i32 = i * blockdim_y;
        s_data[row][column + i * blockdim_y] =
            if (image_h - base_y > shift) d_src[@intCast(offset + shift * pitch)] else 0;
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    inline for (halo_steps..halo_steps + result_steps) |i| {
        var sum: f32 = 0;
        inline for (0..2 * kernel_radius + 1) |j| {
            const y = column + i * blockdim_y + j - kernel_radius;
            sum += c_kernel[2 * kernel_radius - j] * s_data[row][y];
        }
        const shift: i32 = i * blockdim_y;
        d_dst[@intCast(offset + shift * pitch)] = sum;
    }
}
