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

//! An n-body system that runs on one CUDA device.

const std = @import("std");
const cuda = @import("cuda");

pub fn BodySystemCuda(comptime T: type) type {
    return struct {
        const BodySystem = @This();

        num_bodies: u32,
        block_size: u32,
        function: cuda.Function,
        softening_squared: cuda.Buffer(T),
        pos: [2]cuda.Buffer(T),
        vel: cuda.Buffer(T),
        current_read: u1 = 0,
        damping: T = 0.995,

        pub fn init(module: cuda.Module, num_bodies: u32, block_size: u32) !BodySystem {
            const function = try module.function(if (T == f32) "integrateBodiesF32" else "integrateBodiesF64");
            const softening_squared = try module.global(T, if (T == f32) "softening_squared" else "softening_squared_fp64");
            const pos0 = try cuda.alloc(T, num_bodies * 4);
            errdefer pos0.free();
            const pos1 = try cuda.alloc(T, num_bodies * 4);
            errdefer pos1.free();
            const vel = try cuda.alloc(T, num_bodies * 4);
            var body_system: BodySystem = .{
                .num_bodies = num_bodies,
                .block_size = block_size,
                .function = function,
                .softening_squared = softening_squared,
                .pos = .{ pos0, pos1 },
                .vel = vel,
            };
            try body_system.setSoftening(0.00125);
            return body_system;
        }

        pub fn deinit(body_system: BodySystem) void {
            body_system.vel.free();
            body_system.pos[0].free();
            body_system.pos[1].free();
        }

        pub fn setSoftening(body_system: *BodySystem, softening: T) !void {
            try body_system.softening_squared.copyFrom(&.{softening * softening});
        }

        pub fn setDamping(body_system: *BodySystem, damping: T) void {
            body_system.damping = damping;
        }

        pub fn update(body_system: *BodySystem, delta_time: T) !void {
            const block_size = body_system.block_size;
            const num_blocks = (body_system.num_bodies + block_size - 1) / block_size;
            const num_tiles = (body_system.num_bodies + block_size - 1) / block_size;
            const shared_memory_bytes = block_size * 4 * @sizeOf(T);

            const current_read = body_system.current_read;
            try body_system.function.launch(
                .{
                    .grid = .{ .x = num_blocks },
                    .block = .{ .x = block_size },
                    .shared_memory_bytes = shared_memory_bytes,
                },
                .{
                    body_system.pos[1 - current_read],
                    body_system.pos[current_read],
                    body_system.vel,
                    @as(u32, 0),
                    body_system.num_bodies,
                    @as(f32, @floatCast(delta_time)),
                    @as(f32, @floatCast(body_system.damping)),
                    num_tiles,
                },
            );
            body_system.current_read = 1 - current_read;
        }

        /// Copies the positions from the device into `host`.
        pub fn getPositions(body_system: BodySystem, host: []T) !void {
            try body_system.pos[body_system.current_read].copyTo(host);
        }

        pub fn setPositions(body_system: *BodySystem, host: []const T) !void {
            body_system.current_read = 0;
            try body_system.pos[0].copyFrom(host);
        }

        pub fn setVelocities(body_system: BodySystem, host: []const T) !void {
            try body_system.vel.copyFrom(host);
        }
    };
}
