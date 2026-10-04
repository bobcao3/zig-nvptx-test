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

//! Example of program using the interval type and operators: search for roots of a function using
//! an interval Newton method.
//!
//! Use the command-line argument "--n=<N>" to select which GPU implementation to use, otherwise the
//! naive implementation will be used by default.
//! 0: the naive implementation
//! 1: the optimized implementation
//! 2: the recursive implementation

const std = @import("std");
const cuda = @import("cuda");
const constants = @import("interval_constants.zig");
const cpu_interval = @import("cpu_interval.zig");

const I = cpu_interval.I;
const threads = constants.threads;
const depth_result = constants.depth_result;

fn flagValue(args: []const [:0]const u8, name: []const u8) !?i32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') {
            return try std.fmt.parseInt(i32, flag[name.len + 1 ..], 10);
        }
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("[Interval Computing]  starting ...", .{});

    const implementation_choice = try flagValue(args, "n") orelse 0;

    const device = try cuda.init(0);
    defer device.deinit();
    const capability = try device.computeCapability();
    std.log.info("> GPU Device has Compute Capabilities SM {d}.{d}", .{ capability.major, capability.minor });

    switch (implementation_choice) {
        1 => std.log.info("GPU optimized implementation", .{}),
        2 => std.log.info("GPU recursive implementation (requires Compute SM 2.0+)", .{}),
        else => std.log.info("GPU naive implementation", .{}),
    }

    const module = try cuda.Module.load(@embedFile("interval_kernel.ptx"));
    defer module.unload();
    const test_interval_newton = try module.function("testIntervalNewton");

    const d_result = try cuda.alloc(I, threads * depth_result);
    defer d_result.free();
    const d_nresults = try cuda.alloc(i32, threads);
    defer d_nresults.free();
    const h_nresults = try gpa.alloc(i32, threads);
    defer gpa.free(h_nresults);
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    try test_interval_newton.setCacheConfig(.prefer_l1);

    // The non-inlined and recursive calls of the kernel need a deep call stack.
    try cuda.setLimit(.stack_size, 8192);

    const i: I = .init(0.01, 4.0);
    std.log.info("Searching for roots in [{d}, {d}]...", .{ i.lower, i.upper });

    try start.record(.default);

    for (0..constants.num_runs) |_| {
        try test_interval_newton.launch(
            .{ .grid = .{ .x = constants.grid_size }, .block = .{ .x = constants.block_size } },
            .{ d_result, d_nresults, i, implementation_choice },
        );
    }

    try stop.record(.default);
    try cuda.synchronize();

    const h_result = try gpa.alloc(I, threads * depth_result);
    defer gpa.free(h_result);
    try d_result.copyTo(h_result);
    try d_nresults.copyTo(h_nresults);

    std.log.info("Found {d} intervals that may contain the root(s)", .{h_nresults[0]});

    for (0..@intCast(h_nresults[0])) |index| {
        std.log.info(" i[{d}] = [{d:.15}, {d:.15}]", .{
            index,
            h_result[threads * index].lower,
            h_result[threads * index].upper,
        });
    }

    const time = try start.elapsed(stop);
    std.log.info("Number of equations solved: {d}", .{@as(i32, threads)});
    std.log.info("Time per equation: {d} us", .{
        1000000.0 * (time / @as(f32, threads)) / constants.num_runs,
    });

    const h_result_cpu = try gpa.alloc(I, threads * depth_result);
    defer gpa.free(h_result_cpu);
    const h_nresults_cpu = try gpa.alloc(i32, threads);
    defer gpa.free(h_nresults_cpu);
    cpu_interval.testIntervalNewtonCpu(h_result_cpu, h_nresults_cpu, i);

    const test_result = cpu_interval.checkAgainstHost(h_nresults, h_nresults_cpu, h_result, h_result_cpu);
    if (!test_result) std.process.exit(1);
}
