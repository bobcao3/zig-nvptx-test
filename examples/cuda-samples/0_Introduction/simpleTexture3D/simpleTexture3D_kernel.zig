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

//! Renders one slice of a 3D volume into an image using 3D texture lookups.

const std = @import("std");
const nvptx = std.nvptx;

export fn dRender(
    d_output: [*]addrspace(.global) u32,
    image_w: u32,
    image_h: u32,
    w: f32,
    tex_obj: nvptx.TextureObject,
) callconv(.kernel) void {
    const x = nvptx.math.mul24Lo(u32, @workGroupId(0), @workGroupSize(0)) + @workItemId(0);
    const y = nvptx.math.mul24Lo(u32, @workGroupId(1), @workGroupSize(1)) + @workItemId(1);

    const u = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(image_w));
    const v = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(image_h));
    // Read from the 3D texture.
    const voxel = tex_obj.tex(.@"3d", f32, f32, .{ u, v, w })[0];

    if (x < image_w and y < image_h) {
        // Write the output color.
        const i = nvptx.math.mul24Lo(u32, y, image_w) + x;
        d_output[i] = @intFromFloat(voxel * 255);
    }
}
