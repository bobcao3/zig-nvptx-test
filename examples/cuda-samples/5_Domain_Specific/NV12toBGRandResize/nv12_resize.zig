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

//! Implements interlace NV12 frames batch resize.

const std = @import("std");
const cuda = @import("cuda");

pub fn resizeNv12Batch(
    kernel: cuda.Function,
    dp_src: cuda.Buffer(u8),
    n_src_pitch: u32,
    n_src_width: u32,
    n_src_height: u32,
    dp_dst: cuda.Buffer(u8),
    n_dst_pitch: u32,
    n_dst_width: u32,
    n_dst_height: u32,
    n_batch_size: u32,
) !void {
    const hh_src: u32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(n_src_height)) * 3.0 / 2.0));
    var resource: cuda.ResourceDescriptor = .{ .pitch2d = .{
        .address = dp_src.address,
        .format = .unsigned_int8,
        .channels = 1,
        .width = n_src_width,
        .height = hh_src * n_batch_size,
        .pitch_bytes = n_src_pitch,
    } };
    const texture_descriptor: cuda.TextureDescriptor = .{
        .filter_mode = .point,
        .address_mode = @splat(.wrap),
        .read_mode = .element_type,
    };

    const tex_luma = try cuda.TextureObject.create(resource, texture_descriptor);
    defer tex_luma.destroy();

    resource.pitch2d.channels = 2;
    resource.pitch2d.width /= 2;

    const tex_chroma = try cuda.TextureObject.create(resource, texture_descriptor);
    defer tex_chroma.destroy();

    const block: cuda.Dim3 = .{ .x = 32, .y = 32, .z = 1 };

    const block_dim_z = @min(n_batch_size, 32);

    const grid: cuda.Dim3 = .{
        .x = (n_dst_width / 2 + block.x) / block.x,
        .y = (n_dst_height / 2 + block.y) / block.y,
        .z = block_dim_z,
    };
    try kernel.launch(.{ .grid = grid, .block = block }, .{
        tex_luma,
        tex_chroma,
        dp_dst,
        @as(i32, @intCast(n_src_width)),
        @as(i32, @intCast(n_src_height)),
        @as(i32, @intCast(n_dst_pitch)),
        @as(i32, @intCast(n_dst_width)),
        @as(i32, @intCast(n_dst_height)),
        @as(i32, @intCast(n_batch_size)),
    });
}
