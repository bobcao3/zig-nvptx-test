// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

//! Vector addition: C = A + B.
//!
//! This sample is a very basic sample that implements element by element vector addition with
//! unified memory.

const std = @import("std");
const cuda = @import("cuda");

fn initArray(a: []f32, random: std.Random) void {
    for (a) |*element| element.* = random.float(f32);
}

fn serialVecAdd(a: []const f32, b: []const f32, c: []f32) void {
    for (a, b, c) |x, y, *z| z.* = x + y;
}

fn vectorApproximatelyEqual(a: []const f32, b: []const f32, epsilon: f32) bool {
    for (a, b, 0..) |x, y, i| {
        if (@abs(x - y) > epsilon) {
            std.log.err("Index {d} mismatch: {d} != {d}", .{ i, x, y });
            return false;
        }
    }
    return true;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const vector_length = if (args.len >= 2) try std.fmt.parseInt(u31, args[1], 10) else 1024;

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("vectorAdd_kernel.ptx"));
    defer module.unload();
    const vec_add = try module.function("vecAdd");

    const comparison_result = try init.gpa.alloc(f32, vector_length);
    defer init.gpa.free(comparison_result);

    // Unified memory makes A, B and C accessible to the host and the GPU.
    const a = try cuda.allocManaged(f32, vector_length);
    defer a.free();
    const b = try cuda.allocManaged(f32, vector_length);
    defer b.free();
    const c = try cuda.allocManaged(f32, vector_length);
    defer c.free();

    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.real.now(init.io).toSeconds()));
    initArray(a.hostSlice(), prng.random());
    initArray(b.hostSlice(), prng.random());

    const threads = 256;
    const blocks = std.math.divCeil(u32, vector_length, threads) catch unreachable;
    try vec_add.launch(.{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, .{
        a,
        b,
        c,
        @as(i32, vector_length),
    });
    try cuda.synchronize();

    serialVecAdd(a.hostSlice(), b.hostSlice(), comparison_result);

    if (vectorApproximatelyEqual(c.hostSlice(), comparison_result, 0.00001)) {
        std.log.info("Unified Memory: CPU and GPU answers match", .{});
    } else {
        std.log.err("Unified Memory: Error - CPU and GPU answers do not match", .{});
        std.process.exit(1);
    }
}
