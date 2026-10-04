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

//! The kernel that integrates the bodies of an n-body system with all-pairs gravitation.

const std = @import("std");

export var softening_squared: f32 addrspace(.global) = undefined;
export var softening_squared_fp64: f64 addrspace(.global) = undefined;

extern var shared_memory: [0]u8 align(16) addrspace(.shared);

fn NBody(comptime T: type) type {
    return struct {
        const Vec3 = extern struct { x: T, y: T, z: T };
        const Vec4 = extern struct { x: T, y: T, z: T, w: T };

        inline fn getSofteningSquared() T {
            return if (T == f64) softening_squared_fp64 else softening_squared;
        }

        inline fn rsqrt(x: T) T {
            return if (T == f32) std.nvptx.rsqrtApprox(x) else 1.0 / @sqrt(x);
        }

        inline fn bodyBodyInteraction(ai: Vec3, bi: Vec4, bj: Vec4) Vec3 {
            const r: Vec3 = .{ .x = bj.x - bi.x, .y = bj.y - bi.y, .z = bj.z - bi.z };

            var dist_sqr = r.x * r.x + r.y * r.y + r.z * r.z;
            dist_sqr += getSofteningSquared();

            const inv_dist = rsqrt(dist_sqr);
            const inv_dist_cube = inv_dist * inv_dist * inv_dist;

            const s = bj.w * inv_dist_cube;

            return .{ .x = ai.x + r.x * s, .y = ai.y + r.y * s, .z = ai.z + r.z * s };
        }

        inline fn computeBodyAccel(body_pos: Vec4, positions: [*]addrspace(.global) const Vec4, num_tiles: u32) Vec3 {
            const shared_pos: [*]addrspace(.shared) Vec4 = @ptrCast(&shared_memory);
            const work_group_size = @workGroupSize(0);

            var acc: Vec3 = .{ .x = 0, .y = 0, .z = 0 };

            var tile: u32 = 0;
            while (tile < num_tiles) : (tile += 1) {
                shared_pos[@workItemId(0)] = positions[tile * work_group_size + @workItemId(0)];

                std.nvptx.barrierSyncAligned(.work_group, null);

                var counter: u32 = 0;
                while (counter < work_group_size) : (counter += 1) {
                    acc = bodyBodyInteraction(acc, body_pos, shared_pos[counter]);
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            return acc;
        }

        fn integrateBodies(
            new_pos: [*]addrspace(.global) Vec4,
            old_pos: [*]addrspace(.global) const Vec4,
            vel: [*]addrspace(.global) Vec4,
            device_offset: u32,
            device_num_bodies: u32,
            delta_time: f32,
            damping: f32,
            num_tiles: u32,
        ) callconv(.kernel) void {
            const index = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

            if (index >= device_num_bodies) return;

            var position = old_pos[device_offset + index];

            const accel = computeBodyAccel(position, old_pos, num_tiles);

            var velocity = vel[device_offset + index];
            const dt: T = delta_time;
            const damp: T = damping;

            velocity.x += accel.x * dt;
            velocity.y += accel.y * dt;
            velocity.z += accel.z * dt;

            velocity.x *= damp;
            velocity.y *= damp;
            velocity.z *= damp;

            position.x += velocity.x * dt;
            position.y += velocity.y * dt;
            position.z += velocity.z * dt;

            new_pos[device_offset + index] = position;
            vel[device_offset + index] = velocity;
        }

        comptime {
            @export(&integrateBodies, .{ .name = if (T == f32) "integrateBodiesF32" else "integrateBodiesF64" });
        }
    };
}

comptime {
    _ = NBody(f32);
    _ = NBody(f64);
}
