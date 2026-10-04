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

//! Implements NV12 to BGR batch conversion.

const std = @import("std");

inline fn clampF(x: f32, lower: f32, upper: f32) f32 {
    return if (x < lower) lower else if (x > upper) upper else x;
}

inline fn clampedColor(base: [4]f32, offset: [2]f32) @Vector(4, f32) {
    return .{
        clampF(base[0] + offset[0], 0.0, 255.0),
        clampF(base[1] + offset[0], 0.0, 255.0),
        clampF(base[2] + offset[1], 0.0, 255.0),
        clampF(base[3] + offset[1], 0.0, 255.0),
    };
}

pub export fn nv12ToBgrPlanarBatchKernel(
    p_nv12: [*]addrspace(.global) const u8,
    n_nv12_pitch: i32,
    p_bgr: [*]addrspace(.global) f32,
    n_rgb_pitch: i32,
    n_width: i32,
    n_height: i32,
    n_batch_size: i32,
) callconv(.kernel) void {
    const thread_x: i32 = @intCast(@workItemId(0));
    const thread_y: i32 = @intCast(@workItemId(1));
    const x: i32 = @intCast(@workItemId(0) + @workGroupId(0) * @workGroupSize(0));
    const y: i32 = @intCast(@workItemId(1) + @workGroupId(1) * @workGroupSize(1));

    if ((x << 2) + 1 > n_width or (y << 1) + 1 > n_height) return;

    var i: i32 = @intCast(@workGroupId(2));
    while (i < n_batch_size) : (i += @intCast(std.nvptx.nctaid(2))) {
        const p_src = p_nv12 + @as(usize, @intCast(i * ((n_height * n_nv12_pitch * 3) >> 1) +
            (x << 2) + (y << 1) * n_nv12_pitch));
        const luma2x01: [4]u8 = @bitCast(uchar4At(p_src, 0).*);
        const luma2x23: [4]u8 = @bitCast(uchar4At(p_src, n_nv12_pitch).*);
        const uv2: [4]u8 = @bitCast(uchar4At(p_src, (n_height - y) * n_nv12_pitch).*);

        const block_x: i32 = @intCast(@workGroupId(0) * @workGroupSize(0));
        const block_y: i32 = @intCast(@workGroupId(1) * @workGroupSize(1));
        const p_dst_block = p_bgr + @as(usize, @intCast(i * ((n_height * n_rgb_pitch * 3) >> 2) +
            (block_x << 2) + (block_y << 1) * (n_rgb_pitch >> 2)));

        const add00: [2]f32 = .{ 1.1644 * f(luma2x01[0]), 1.1644 * f(luma2x01[2]) };
        const add01: [2]f32 = .{ 1.1644 * f(luma2x01[1]), 1.1644 * f(luma2x01[3]) };
        const add02: [2]f32 = .{ 1.1644 * f(luma2x23[0]), 1.1644 * f(luma2x23[2]) };
        const add03: [2]f32 = .{ 1.1644 * f(luma2x23[1]), 1.1644 * f(luma2x23[3]) };

        const d: [2]f32 = .{ f(uv2[0]) - 128.0, f(uv2[2]) - 128.0 };
        const e: [2]f32 = .{ f(uv2[1]) - 128.0, f(uv2[3]) - 128.0 };

        const add1: [2]f32 = .{ 2.0172 * d[0], 2.0172 * d[1] };

        const add2: [2]f32 = .{
            (-0.3918) * d[0] + (-0.8130) * e[0],
            (-0.3918) * d[1] + (-0.8130) * e[1],
        };

        const add3: [2]f32 = .{ 1.5960 * e[0], 1.5960 * e[1] };

        const row_stride = (thread_y << 1) * (n_rgb_pitch >> 2);
        const next_row_stride = ((thread_y << 1) + 1) * (n_rgb_pitch >> 2);
        const plane_stride = (n_height * n_rgb_pitch) >> 2;
        const column = thread_x << 2;
        const rows = [2]i32{ row_stride, next_row_stride };
        const lumas = [2][4]f32{
            .{ add00[0], add01[0], add00[1], add01[1] },
            .{ add02[0], add03[0], add02[1], add03[1] },
        };
        const planes = [3][2]f32{ add1, add2, add3 };
        inline for (planes, 0..) |chroma, plane| {
            inline for (rows, lumas) |row, luma| {
                float4At(p_dst_block, plane_stride * @as(i32, plane) + row + column).* =
                    clampedColor(luma, chroma);
            }
        }
    }
}

inline fn uchar4At(
    base: [*]addrspace(.global) const u8,
    offset: i32,
) *align(4) addrspace(.global) const u32 {
    return @ptrCast(@alignCast(base + @as(usize, @intCast(offset))));
}

inline fn float4At(
    base: [*]addrspace(.global) f32,
    offset: i32,
) *align(16) addrspace(.global) @Vector(4, f32) {
    return @ptrCast(@alignCast(base + @as(usize, @intCast(offset))));
}

inline fn f(byte: u8) f32 {
    return @floatFromInt(byte);
}
