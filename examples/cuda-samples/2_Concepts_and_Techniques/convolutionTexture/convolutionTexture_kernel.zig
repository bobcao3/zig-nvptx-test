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

//! This sample implements the same algorithm as the convolutionSeparable
//! CUDA Sample, but without using the shared memory at all.
//! Instead, it uses textures in exactly the same way an OpenGL-based
//! implementation would do.
//! Refer to the "Performance" section of convolutionSeparable whitepaper.

const std = @import("std");
const common = @import("convolutionTexture_common.zig");

const kernel_radius = common.kernel_radius;

inline fn imad(a: i32, b: i32, c: i32) i32 {
    return std.nvptx.math.mul24Lo(i32, a, b) + c;
}

inline fn globalIndex(comptime dimension: u32) i32 {
    const block_size: i32 = @intCast(@workGroupSize(dimension));
    return imad(block_size, @intCast(@workGroupId(dimension)), @intCast(@workItemId(dimension)));
}

export var c_kernel: [common.kernel_length]f32 addrspace(.global) = undefined;

inline fn tap(comptime i: usize) f32 {
    return @floatFromInt(@as(i32, kernel_radius) - @as(i32, i));
}

inline fn convolutionRow(x: f32, y: f32, tex_src: std.nvptx.TextureObject) f32 {
    var sum: f32 = 0;
    inline for (0..2 * kernel_radius + 1) |i| {
        const position: @Vector(2, f32) = .{ x + tap(i), y };
        sum = tex_src.tex(.@"2d", f32, f32, position)[0] * c_kernel[i] + sum;
    }
    return sum;
}

inline fn convolutionColumn(x: f32, y: f32, tex_src: std.nvptx.TextureObject) f32 {
    var sum: f32 = 0;
    inline for (0..2 * kernel_radius + 1) |i| {
        const position: @Vector(2, f32) = .{ x, y + tap(i) };
        sum = tex_src.tex(.@"2d", f32, f32, position)[0] * c_kernel[i] + sum;
    }
    return sum;
}

export fn convolutionRowsKernel(
    d_dst: [*]addrspace(.global) f32,
    image_w: i32,
    image_h: i32,
    tex_src: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const ix = globalIndex(0);
    const iy = globalIndex(1);
    const x = @as(f32, @floatFromInt(ix)) + 0.5;
    const y = @as(f32, @floatFromInt(iy)) + 0.5;

    if (ix >= image_w or iy >= image_h) return;

    const sum = convolutionRow(x, y, tex_src);

    d_dst[@intCast(imad(iy, image_w, ix))] = sum;
}

export fn convolutionColumnsKernel(
    d_dst: [*]addrspace(.global) f32,
    image_w: i32,
    image_h: i32,
    tex_src: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const ix = globalIndex(0);
    const iy = globalIndex(1);
    const x = @as(f32, @floatFromInt(ix)) + 0.5;
    const y = @as(f32, @floatFromInt(iy)) + 0.5;

    if (ix >= image_w or iy >= image_h) return;

    const sum = convolutionColumn(x, y, tex_src);

    d_dst[@intCast(imad(iy, image_w, ix))] = sum;
}
