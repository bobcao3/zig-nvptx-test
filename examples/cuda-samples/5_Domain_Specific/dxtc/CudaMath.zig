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

//! Math functions of the DXT compressor for vectors of three floats.

const std = @import("std");

pub const F3 = @Vector(3, f32);

pub inline fn splat3(x: f32) F3 {
    return @splat(x);
}

pub inline fn dot(a: F3, b: F3) f32 {
    return @reduce(.Add, a * b);
}

/// A group of 16 work items of the first warp, which synchronizes with `bar.warp.sync`.
pub const Tile = struct {
    member_mask: std.nvptx.MemberMask,

    pub inline fn sync(tile: Tile) void {
        std.nvptx.barWarpSync(tile.member_mask);
    }
};

/// Uses the power method to find the first eigenvector.
/// https://en.wikipedia.org/wiki/Power_iteration
pub inline fn firstEigenVector(matrix: [*]addrspace(.shared) const f32) F3 {
    // 8 iterations seems to be more than enough.
    var v: F3 = .{ 1.0, 1.0, 1.0 };

    for (0..8) |_| {
        const x = v[0] * matrix[0] + v[1] * matrix[1] + v[2] * matrix[2];
        const y = v[0] * matrix[1] + v[1] * matrix[3] + v[2] * matrix[4];
        const z = v[0] * matrix[2] + v[1] * matrix[4] + v[2] * matrix[5];
        const m = @max(@max(x, y), z);
        const iv = 1.0 / m;
        v = .{ x * iv, y * iv, z * iv };
    }

    return v;
}

pub inline fn colorSums(
    colors: *addrspace(.shared) const [16]F3,
    sums: *addrspace(.shared) [16]F3,
    tile: Tile,
) void {
    const idx = @workItemId(0);

    sums[idx] = colors[idx];
    tile.sync();
    sums[idx] += sums[idx ^ 8];
    tile.sync();
    sums[idx] += sums[idx ^ 4];
    tile.sync();
    sums[idx] += sums[idx ^ 2];
    tile.sync();
    sums[idx] += sums[idx ^ 1];
}

var covariance: [16 * 6]f32 addrspace(.shared) = undefined;

pub inline fn bestFitLine(
    colors: *addrspace(.shared) const [16]F3,
    color_sum: F3,
    tile: Tile,
) F3 {
    // Compute the covariance matrix of the given colors.
    const idx = @workItemId(0);

    const diff = colors[idx] - color_sum * splat3(1.0 / 16.0);

    covariance[6 * idx + 0] = diff[0] * diff[0];
    covariance[6 * idx + 1] = diff[0] * diff[1];
    covariance[6 * idx + 2] = diff[0] * diff[2];
    covariance[6 * idx + 3] = diff[1] * diff[1];
    covariance[6 * idx + 4] = diff[1] * diff[2];
    covariance[6 * idx + 5] = diff[2] * diff[2];

    tile.sync();
    var d: u32 = 8;
    while (d > 0) : (d >>= 1) {
        if (idx < d) {
            inline for (0..6) |component| {
                covariance[6 * idx + component] += covariance[6 * (idx + d) + component];
            }
        }
        tile.sync();
    }

    // Compute the first eigenvector.
    return firstEigenVector(&covariance);
}
