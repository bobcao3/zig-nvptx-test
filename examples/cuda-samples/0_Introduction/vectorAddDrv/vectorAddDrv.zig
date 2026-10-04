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

//! Vector addition: C = A + B.
//!
//! This sample is a very basic sample that implements element by element vector addition. It is
//! the same as the sample illustrating Chapter 3 of the programming guide with some additions
//! like error checking.

const std = @import("std");
const cuda = @import("cuda");

fn randomInit(data: []f32, random: std.Random) void {
    for (data) |*element| element.* = random.float(f32);
}

pub fn main(init: std.process.Init) !void {
    std.log.info("Vector Addition (Driver API)", .{});
    const n = 50000;

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("vectorAdd_kernel.ptx"));
    defer module.unload();
    const vec_add_kernel = try module.function("vecAddKernel");

    const h_a = try init.gpa.alloc(f32, n);
    defer init.gpa.free(h_a);
    const h_b = try init.gpa.alloc(f32, n);
    defer init.gpa.free(h_b);
    const h_c = try init.gpa.alloc(f32, n);
    defer init.gpa.free(h_c);

    var prng: std.Random.DefaultPrng = .init(2006);
    randomInit(h_a, prng.random());
    randomInit(h_b, prng.random());

    const d_a = try cuda.alloc(f32, n);
    defer d_a.free();
    const d_b = try cuda.alloc(f32, n);
    defer d_b.free();
    const d_c = try cuda.alloc(f32, n);
    defer d_c.free();

    try d_a.copyFrom(h_a);
    try d_b.copyFrom(h_b);

    const threads_per_block = 256;
    const blocks_per_grid = std.math.divCeil(u32, n, threads_per_block) catch unreachable;
    try vec_add_kernel.launch(
        .{ .grid = .{ .x = blocks_per_grid }, .block = .{ .x = threads_per_block } },
        .{ d_a, d_b, d_c, @as(i32, n) },
    );

    try d_c.copyTo(h_c);

    const matches = for (h_a, h_b, h_c) |a, b, c| {
        if (@abs(c - (a + b)) > 1e-7) break false;
    } else true;
    if (matches) std.log.info("Result = PASS", .{}) else std.log.err("Result = FAIL", .{});
    if (!matches) std.process.exit(1);
}
