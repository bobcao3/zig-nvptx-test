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
const cuda = @import("cuda");

pub const Crop = struct {
    x: u32 = 0,
    y: u32 = 0,
    w: u32 = 0,
    h: u32 = 0,
};

fn gridExtent(crop_extent: u32, scale: f32, block_extent: u32) u32 {
    const block_extent_f: f32 = @floatFromInt(block_extent);
    const blocks = (@as(f32, @floatFromInt(crop_extent)) * 1.0 / scale + block_extent_f - 1) /
        block_extent_f;
    return @intFromFloat(blocks);
}

pub fn resizeBgrPlanarBatch(
    kernel: cuda.Function,
    dp_src: cuda.Buffer(f32),
    n_src_pitch: u32,
    n_src_width: u32,
    n_src_height: u32,
    dp_dst: cuda.Buffer(f32),
    n_dst_pitch: u32,
    n_dst_width: u32,
    n_dst_height: u32,
    n_batch_size: u32,
    crop_in: Crop,
    wh_same_resize_ratio: bool,
) !void {
    var tex_src: [2]cuda.TextureObject = undefined;
    var created_textures: usize = 0;
    defer for (tex_src[0..created_textures]) |texture| texture.destroy();
    var n_tiles: u32 = 1;

    const h = n_src_height * 3 * n_batch_size;
    while ((h + n_tiles - 1) / n_tiles > 65536) n_tiles += 1;

    if (n_tiles > 2) return;

    const batch_tile = n_batch_size / n_tiles;
    const batch_tile_last = n_batch_size - batch_tile * (n_tiles - 1);

    var crop = crop_in;
    for (0..n_tiles) |i_tile| {
        const bs = if (i_tile == n_tiles - 1) batch_tile_last else batch_tile;
        const src_offset = i_tile * (batch_tile * 3 * n_src_height * n_src_pitch);
        const dp_src_new: cuda.Buffer(f32) = .{
            .address = dp_src.address + src_offset * @sizeOf(f32),
            .len = dp_src.len - src_offset,
        };

        tex_src[i_tile] = try cuda.TextureObject.create(.{ .pitch2d = .{
            .address = dp_src_new.address,
            .format = .float,
            .channels = 1,
            .width = n_src_width,
            .height = bs * 3 * n_src_height,
            .pitch_bytes = n_src_pitch * @sizeOf(f32),
        } }, .{
            .filter_mode = .linear,
            .address_mode = @splat(.wrap),
            .read_mode = .element_type,
        });
        created_textures += 1;
        const dst_offset = i_tile * (batch_tile * 3 * n_dst_height * n_dst_pitch);
        const dp_dst_new: cuda.Buffer(f32) = .{
            .address = dp_dst.address + dst_offset * @sizeOf(f32),
            .len = dp_dst.len - dst_offset,
        };

        if (crop.w == 0 or crop.h == 0) {
            crop = .{ .x = 0, .y = 0, .w = n_src_width, .h = n_src_height };
        }

        var scale_x = @as(f32, @floatFromInt(crop.w)) * 1.0 / @as(f32, @floatFromInt(n_dst_width));
        var scale_y = @as(f32, @floatFromInt(crop.h)) * 1.0 / @as(f32, @floatFromInt(n_dst_height));

        if (wh_same_resize_ratio) {
            scale_x = if (scale_x > scale_y) scale_x else scale_y;
            scale_y = scale_x;
        }
        const block: cuda.Dim3 = .{ .x = 32, .y = 32, .z = 1 };

        const block_dim_z = @min(bs, 32);
        const grid: cuda.Dim3 = .{
            .x = gridExtent(crop.w, scale_x, block.x),
            .y = gridExtent(crop.h, scale_y, block.y),
            .z = block_dim_z,
        };

        try kernel.launch(.{ .grid = grid, .block = block }, .{
            tex_src[i_tile],
            dp_dst_new,
            @as(i32, @intCast(n_dst_pitch)),
            @as(i32, @intCast(n_dst_height)),
            @as(i32, @intCast(n_src_height)),
            @as(i32, @intCast(bs)),
            scale_x,
            scale_y,
            @as(i32, @intCast(crop.x)),
            @as(i32, @intCast(crop.y)),
            @as(i32, @intCast(crop.w)),
            @as(i32, @intCast(crop.h)),
        });
    }
}
