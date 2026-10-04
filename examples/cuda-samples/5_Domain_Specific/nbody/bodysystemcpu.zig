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

//! An n-body system that runs on the CPU.

const std = @import("std");

pub fn BodySystemCpu(comptime T: type) type {
    return struct {
        const BodySystem = @This();

        num_bodies: usize,
        pos: []T,
        vel: []T,
        force: []T,
        softening_squared: T = 0.00125,
        damping: T = 0.995,

        pub fn init(gpa: std.mem.Allocator, num_bodies: usize) !BodySystem {
            const pos = try gpa.alloc(T, num_bodies * 4);
            errdefer gpa.free(pos);
            const vel = try gpa.alloc(T, num_bodies * 4);
            errdefer gpa.free(vel);
            const force = try gpa.alloc(T, num_bodies * 3);
            @memset(pos, 0);
            @memset(vel, 0);
            @memset(force, 0);
            return .{ .num_bodies = num_bodies, .pos = pos, .vel = vel, .force = force };
        }

        pub fn deinit(body_system: BodySystem, gpa: std.mem.Allocator) void {
            gpa.free(body_system.pos);
            gpa.free(body_system.vel);
            gpa.free(body_system.force);
        }

        pub fn update(body_system: *BodySystem, delta_time: T) void {
            body_system.integrateNBodySystem(delta_time);
        }

        fn bodyBodyInteraction(accel: *[3]T, pos_mass0: []const T, pos_mass1: []const T, softening_squared: T) void {
            var r: [3]T = undefined;
            r[0] = pos_mass1[0] - pos_mass0[0];
            r[1] = pos_mass1[1] - pos_mass0[1];
            r[2] = pos_mass1[2] - pos_mass0[2];

            var dist_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            dist_sqr += softening_squared;

            const inv_dist: T = @floatCast(1.0 / @sqrt(@as(f64, dist_sqr)));
            const inv_dist_cube = inv_dist * inv_dist * inv_dist;

            const s = pos_mass1[3] * inv_dist_cube;

            accel[0] += r[0] * s;
            accel[1] += r[1] * s;
            accel[2] += r[2] * s;
        }

        fn computeNBodyGravitation(body_system: *BodySystem) void {
            for (0..body_system.num_bodies) |i| {
                var acc: [3]T = .{ 0, 0, 0 };

                for (0..body_system.num_bodies) |j| {
                    bodyBodyInteraction(
                        &acc,
                        body_system.pos[4 * i ..][0..4],
                        body_system.pos[4 * j ..][0..4],
                        body_system.softening_squared,
                    );
                }

                body_system.force[3 * i ..][0..3].* = acc;
            }
        }

        fn integrateNBodySystem(body_system: *BodySystem, delta_time: T) void {
            body_system.computeNBodyGravitation();

            for (0..body_system.num_bodies) |i| {
                const index = 4 * i;
                const index_force = 3 * i;

                var pos: [3]T = body_system.pos[index..][0..3].*;
                var vel: [3]T = body_system.vel[index..][0..3].*;
                const inv_mass = body_system.pos[index + 3];
                const force = body_system.force[index_force..][0..3].*;

                for (&vel, force) |*v, f| {
                    v.* += (f * inv_mass) * delta_time;
                    v.* *= body_system.damping;
                }

                for (&pos, vel) |*p, v| p.* += v * delta_time;

                body_system.pos[index..][0..3].* = pos;
                body_system.vel[index..][0..3].* = vel;
            }
        }
    };
}
