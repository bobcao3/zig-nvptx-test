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

//! Shifts matrix elements with wrap addressing, once through a texture bound to pitch linear
//! memory and once through a texture bound to a CUDA array.
//!
//! The pitch is in elements, not bytes. The textures use normalized coordinates, which the wrap
//! address mode requires.

const std = @import("std");
const nvptx = std.nvptx;

inline fn shift(
    odata: [*]addrspace(.global) f32,
    pitch: i32,
    width: i32,
    height: i32,
    shift_x: i32,
    shift_y: i32,
    texture: nvptx.TextureObject,
) void {
    const xid: i32 = @intCast(@workGroupId(0) * @workGroupSize(0) + @workItemId(0));
    const yid: i32 = @intCast(@workGroupId(1) * @workGroupSize(1) + @workItemId(1));

    const u = @as(f32, @floatFromInt(xid + shift_x)) / @as(f32, @floatFromInt(width));
    const v = @as(f32, @floatFromInt(yid + shift_y)) / @as(f32, @floatFromInt(height));
    odata[@intCast(yid * pitch + xid)] = texture.tex(.@"2d", f32, f32, .{ u, v })[0];
}

/// Shifts matrix elements using pitch linear memory.
export fn shiftPitchLinear(
    odata: [*]addrspace(.global) f32,
    pitch: i32,
    width: i32,
    height: i32,
    shift_x: i32,
    shift_y: i32,
    tex_ref_pl: nvptx.TextureObject,
) callconv(.kernel) void {
    shift(odata, pitch, width, height, shift_x, shift_y, tex_ref_pl);
}

/// Shifts matrix elements using a regular array.
export fn shiftArray(
    odata: [*]addrspace(.global) f32,
    pitch: i32,
    width: i32,
    height: i32,
    shift_x: i32,
    shift_y: i32,
    tex_ref_array: nvptx.TextureObject,
) callconv(.kernel) void {
    shift(odata, pitch, width, height, shift_x, shift_y, tex_ref_array);
}
