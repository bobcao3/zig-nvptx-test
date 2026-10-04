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

//! This sample demonstrates how use texture fetches in CUDA
//!
//! This sample takes an input PGM image (imageFilename) and generates
//! an output PGM image (imageFilename_out).  This CUDA kernel performs
//! a simple 2D transform (rotation) on the texture coordinates (u,v).

const std = @import("std");

/// Writes to a cuArray (texture data source) using surface writes.
export fn surfaceWriteKernel(
    input: [*]addrspace(.global) const f32,
    width: i32,
    height: i32,
    output_surface: std.nvptx.SurfaceObject,
) callconv(.kernel) void {
    _ = height;
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);

    output_surface.sustB(.@"2d", .trap, .{ @intCast(x * 4), @intCast(y) }, @Vector(1, f32){
        input[y * @as(u32, @intCast(width)) + x],
    });
}

/// Transforms an image using texture lookups.
export fn transformKernel(
    output: [*]addrspace(.global) f32,
    width: i32,
    height: i32,
    theta: f32,
    tex: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);

    var u = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(width));
    var v = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(height));

    u -= 0.5;
    v -= 0.5;
    const tu = u * std.nvptx.cosApprox(theta) - v * std.nvptx.sinApprox(theta) + 0.5;
    const tv = v * std.nvptx.cosApprox(theta) + u * std.nvptx.sinApprox(theta) + 0.5;

    output[y * @as(u32, @intCast(width)) + x] = tex.tex(.@"2d", f32, f32, .{ tu, tv })[0];
}
