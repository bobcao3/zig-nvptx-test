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

pub export fn resizeNv12BatchKernel(
    tex_src_luma: std.nvptx.TextureObject,
    tex_src_chroma: std.nvptx.TextureObject,
    p_dst_nv12: [*]addrspace(.global) u8,
    n_src_width: i32,
    n_src_height: i32,
    n_dst_pitch: i32,
    n_dst_width: i32,
    n_dst_height: i32,
    n_batch_size: i32,
) callconv(.kernel) void {
    const x: i32 = @intCast(@workItemId(0) + @workGroupId(0) * @workGroupSize(0));
    const y: i32 = @intCast(@workItemId(1) + @workGroupId(1) * @workGroupSize(1));

    const px = x * 2;
    var py = y * 2;

    if ((px + 1) >= n_dst_width or (py + 1) >= n_dst_height) return;

    const fx_scale = 1.0 * floatOf(n_src_width) / floatOf(n_dst_width);
    const fy_scale = 1.0 * floatOf(n_src_height) / floatOf(n_dst_height);

    var p = p_dst_nv12 + @as(usize, @intCast(px + py * n_dst_pitch));
    const hh = @divTrunc(n_dst_height * 3, 2);
    const n_byte = n_dst_pitch * hh;
    const px_fx_scale: i32 = @intFromFloat(floatOf(px) * fx_scale);
    const px_fx_scale_1: i32 = @intFromFloat(floatOf(px + 1) * fx_scale);
    const py_fy_scale: i32 = @intFromFloat(floatOf(py) * fy_scale);
    const py_fy_scale_1: i32 = @intFromFloat(floatOf(py + 1) * fy_scale);

    var i: i32 = @intCast(@workGroupId(2));
    while (i < n_batch_size) : (i += @intCast(std.nvptx.nctaid(2))) {
        uchar2At(p, 0).* = .{
            lumaTexel(tex_src_luma, px_fx_scale, py_fy_scale),
            lumaTexel(tex_src_luma, px_fx_scale_1, py_fy_scale),
        };
        uchar2At(p, n_dst_pitch).* = .{
            lumaTexel(tex_src_luma, px_fx_scale, py_fy_scale_1),
            lumaTexel(tex_src_luma, px_fx_scale_1, py_fy_scale_1),
        };
        const chroma_x = floatOf(x) * fx_scale;
        const chroma_y = floatOf(hh * i + n_dst_height + y) * fy_scale;
        const texel = tex_src_chroma.tex(.@"2d", u32, f32, .{ chroma_x, chroma_y });
        const chroma = uchar2At(p, (n_dst_height - y) * n_dst_pitch);
        chroma.* = .{ @truncate(texel[0]), @truncate(texel[1]) };
        p += @as(usize, @intCast(n_byte));
        py += hh;
    }
}

inline fn floatOf(value: i32) f32 {
    return @floatFromInt(value);
}

inline fn uchar2At(base: [*]addrspace(.global) u8, offset: i32) *align(2) addrspace(.global) [2]u8 {
    return @ptrCast(@alignCast(base + @as(usize, @intCast(offset))));
}

inline fn lumaTexel(texture: std.nvptx.TextureObject, x: i32, y: i32) u8 {
    const texel = texture.tex(.@"2d", u32, f32, .{ @floatFromInt(x), @floatFromInt(y) });
    return @truncate(texel[0]);
}
