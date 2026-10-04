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

//! Transforms an image using texture lookups: a rotation of the texture coordinates.

const std = @import("std");
const nvptx = std.nvptx;

export fn transformKernel(
    output_data: [*]addrspace(.global) f32,
    width: i32,
    height: i32,
    theta: f32,
    tex: nvptx.TextureObject,
) callconv(.kernel) void {
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);

    const w: f32 = @floatFromInt(width);
    const h: f32 = @floatFromInt(height);
    const u = @as(f32, @floatFromInt(x)) - w / 2;
    const v = @as(f32, @floatFromInt(y)) - h / 2;
    var tu = u * nvptx.cosApprox(theta) - v * nvptx.sinApprox(theta);
    var tv = v * nvptx.cosApprox(theta) + u * nvptx.sinApprox(theta);

    tu /= w;
    tv /= h;

    output_data[y * @as(u32, @intCast(width)) + x] = tex.tex(.@"2d", f32, f32, .{ tu + 0.5, tv + 0.5 })[0];
}
