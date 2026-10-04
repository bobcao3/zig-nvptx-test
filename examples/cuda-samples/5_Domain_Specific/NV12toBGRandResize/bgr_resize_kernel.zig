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

//! Implements BGR 3 progressive planars frames batch resize.

const std = @import("std");

pub export fn resizeBgrPlanarBatchKernel(
    tex_src: std.nvptx.TextureObject,
    p_dst: [*]addrspace(.global) f32,
    n_dst_pitch: i32,
    n_dst_height: i32,
    n_src_height: i32,
    batch: i32,
    scale_x: f32,
    scale_y: f32,
    crop_x: i32,
    crop_y: i32,
    crop_w: i32,
    crop_h: i32,
) callconv(.kernel) void {
    const x: i32 = @intCast(@workItemId(0) + @workGroupId(0) * @workGroupSize(0));
    const y: i32 = @intCast(@workItemId(1) + @workGroupId(1) * @workGroupSize(1));

    const width_limit: i32 = @intFromFloat(@as(f32, @floatFromInt(crop_w)) / scale_x);
    const height_limit: i32 = @intFromFloat(@as(f32, @floatFromInt(crop_h)) / scale_y);
    if (x >= width_limit or y >= height_limit) return;

    const frame_size = n_dst_pitch * n_dst_height;
    var i: i32 = @intCast(@workGroupId(2));
    while (i < batch) : (i += @intCast(std.nvptx.nctaid(2))) {
        inline for (0..3) |channel| {
            const plane: i32 = channel;
            const offset = i * 3 * frame_size + y * n_dst_pitch + x + plane * frame_size;
            const p = p_dst + @as(usize, @intCast(offset));
            const coordinate_x = @as(f32, @floatFromInt(x)) * scale_x +
                @as(f32, @floatFromInt(crop_x));
            const coordinate_y = @as(f32, @floatFromInt((3 * i + plane) * n_src_height)) +
                @as(f32, @floatFromInt(y)) * scale_y + @as(f32, @floatFromInt(crop_y));
            p[0] = tex_src.tex(.@"2d", f32, f32, .{ coordinate_x, coordinate_y })[0];
        }
    }
}
