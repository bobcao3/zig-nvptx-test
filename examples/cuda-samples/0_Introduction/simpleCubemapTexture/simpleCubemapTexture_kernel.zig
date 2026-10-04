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

//! Transforms the faces of a cubemap into a linear buffer using cubemap texture lookups.

const std = @import("std");
const nvptx = std.nvptx;

export fn transformKernel(
    g_odata: [*]addrspace(.global) f32,
    width: i32,
    tex: nvptx.TextureObject,
) callconv(.kernel) void {
    // Calculate the data point of this work item.
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);

    // The 0.5 offset and the division access the original data points in the texture, so that
    // bilinear interpolation does not take effect.
    const w: f32 = @floatFromInt(width);
    const u = ((@as(f32, @floatFromInt(x)) + 0.5) / w) * 2.0 - 1.0;
    const v = ((@as(f32, @floatFromInt(y)) + 0.5) / w) * 2.0 - 1.0;

    const w_items: u32 = @intCast(width);
    for (0..6) |face| {
        const direction: @Vector(3, f32) = switch (face) {
            // Layer 0 is the positive X face.
            0 => .{ 1, -v, -u },
            // Layer 1 is the negative X face.
            1 => .{ -1, -v, u },
            // Layer 2 is the positive Y face.
            2 => .{ u, 1, v },
            // Layer 3 is the negative Y face.
            3 => .{ u, -1, -v },
            // Layer 4 is the positive Z face.
            4 => .{ u, -v, 1 },
            // Layer 5 is the negative Z face.
            5 => .{ -u, -v, -1 },
            else => unreachable,
        };

        // Read from the texture, do the expected transformation and write to global memory.
        g_odata[face * w_items * w_items + y * w_items + x] = -tex.tex(.cube, f32, f32, direction)[0];
    }
}
