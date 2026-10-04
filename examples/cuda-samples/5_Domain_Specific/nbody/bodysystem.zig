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

//! Initial conditions of an n-body system.

const std = @import("std");

pub const Float3 = @Vector(3, f32);

fn dot(v0: Float3, v1: Float3) f32 {
    return @reduce(.Add, v0 * v1);
}

fn cross(v0: Float3, v1: Float3) Float3 {
    return .{
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
    };
}

fn normalize(vector: *Float3) f32 {
    const dist = @sqrt(dot(vector.*, vector.*));
    if (dist > 1e-6) vector.* /= @splat(dist);
    return dist;
}

/// Places the bodies on a spherical shell. `pos` and `vel` hold four values per body.
pub fn randomizeBodies(
    comptime T: type,
    random: std.Random,
    pos: []T,
    vel: []T,
    cluster_scale: f32,
    velocity_scale: f32,
    num_bodies: usize,
) void {
    const scale = cluster_scale;
    const vscale = scale * velocity_scale;
    const inner = 2.5 * scale;
    const outer = 4.0 * scale;

    var i: usize = 0;
    while (i < num_bodies) {
        const x = random.float(f32) * 2 - 1;
        const y = random.float(f32) * 2 - 1;
        const z = random.float(f32) * 2 - 1;
        var point: Float3 = .{ x, y, z };
        const len = normalize(&point);
        if (len > 1) continue;

        pos[4 * i + 0] = point[0] * (inner + (outer - inner) * random.float(f32));
        pos[4 * i + 1] = point[1] * (inner + (outer - inner) * random.float(f32));
        pos[4 * i + 2] = point[2] * (inner + (outer - inner) * random.float(f32));
        pos[4 * i + 3] = 1.0;

        var axis: Float3 = .{ 0, 0, 1 };
        _ = normalize(&axis);
        if (1 - dot(point, axis) < 1e-6) {
            axis[0] = point[1];
            axis[1] = point[0];
            _ = normalize(&axis);
        }
        var vv: Float3 = .{ @floatCast(pos[4 * i]), @floatCast(pos[4 * i + 1]), @floatCast(pos[4 * i + 2]) };
        vv = cross(vv, axis);
        vel[4 * i + 0] = vv[0] * vscale;
        vel[4 * i + 1] = vv[1] * vscale;
        vel[4 * i + 2] = vv[2] * vscale;
        vel[4 * i + 3] = 1.0;
        i += 1;
    }
}
