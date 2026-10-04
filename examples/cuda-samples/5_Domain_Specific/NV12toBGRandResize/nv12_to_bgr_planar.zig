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
const cuda = @import("cuda");

const conv_threads_x = 64;
const conv_threads_y = 10;

pub fn nv12ToBgrPlanarBatch(
    kernel: cuda.Function,
    p_nv12: cuda.Buffer(u8),
    n_nv12_pitch: u32,
    p_bgr: cuda.Buffer(f32),
    n_rgb_pitch: u32,
    n_width: u32,
    n_height: u32,
    n_batch_size: u32,
) !void {
    const threads: cuda.Dim3 = .{ .x = conv_threads_x, .y = conv_threads_y };

    const block_dim_z = @min(n_batch_size, 32);

    const blocks: cuda.Dim3 = .{
        .x = (n_width / 4 - 1) / threads.x + 1,
        .y = (n_height / 2 - 1) / threads.y + 1,
        .z = block_dim_z,
    };
    try kernel.launch(.{ .grid = blocks, .block = threads }, .{
        p_nv12,
        @as(i32, @intCast(n_nv12_pitch)),
        p_bgr,
        @as(i32, @intCast(n_rgb_pitch)),
        @as(i32, @intCast(n_width)),
        @as(i32, @intCast(n_height)),
        @as(i32, @intCast(n_batch_size)),
    });
}
