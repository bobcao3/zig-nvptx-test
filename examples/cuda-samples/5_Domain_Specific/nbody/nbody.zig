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

//! A gravitational n-body simulation. This sample accompanies the GPU Gems 3 chapter "Fast N-Body
//! Simulation with CUDA". It runs the benchmark mode and the mode that compares the results of
//! the GPU with the results of the CPU, and has no OpenGL display. Without `-benchmark`, it
//! compares.

const std = @import("std");
const cuda = @import("cuda");
const bodysystem = @import("bodysystem.zig");
const BodySystemCpu = @import("bodysystemcpu.zig").BodySystemCpu;
const BodySystemCuda = @import("bodysystemcuda.zig").BodySystemCuda;

const Params = struct {
    timestep: f32,
    cluster_scale: f32,
    velocity_scale: f32,
    softening: f32,
    damping: f32,
};

var active_params: Params = .{
    .timestep = 0.016,
    .cluster_scale = 1.54,
    .velocity_scale = 8.0,
    .softening = 0.1,
    .damping = 1.0,
};

fn flopsPerInteraction(comptime T: type) comptime_int {
    return if (T == f64) 30 else 20;
}

const PerfStats = struct { interactions_per_second: f64, gflops: f64 };

fn computePerfStats(comptime T: type, num_bodies: u32, milliseconds: f32, iterations: u32) PerfStats {
    var interactions_per_second: f64 = @as(f32, @floatFromInt(num_bodies)) * @as(f32, @floatFromInt(num_bodies));
    interactions_per_second *= 1e-9 * @as(f64, @floatFromInt(iterations)) * 1000 / milliseconds;
    return .{
        .interactions_per_second = interactions_per_second,
        .gflops = interactions_per_second * @as(f32, flopsPerInteraction(T)),
    };
}

fn NBodyDemo(comptime T: type) type {
    return struct {
        const Demo = @This();

        gpa: std.mem.Allocator,
        num_bodies: u32,
        nbody_cuda: BodySystemCuda(T),
        pos: []T,
        vel: []T,

        fn init(gpa: std.mem.Allocator, module: cuda.Module, num_bodies: u32, block_size: u32) !Demo {
            var nbody_cuda: BodySystemCuda(T) = try .init(module, num_bodies, block_size);
            errdefer nbody_cuda.deinit();
            const pos = try gpa.alloc(T, num_bodies * 4);
            errdefer gpa.free(pos);
            const vel = try gpa.alloc(T, num_bodies * 4);
            errdefer gpa.free(vel);

            try nbody_cuda.setSoftening(active_params.softening);
            nbody_cuda.setDamping(active_params.damping);

            return .{ .gpa = gpa, .num_bodies = num_bodies, .nbody_cuda = nbody_cuda, .pos = pos, .vel = vel };
        }

        fn deinit(demo: Demo) void {
            demo.nbody_cuda.deinit();
            demo.gpa.free(demo.pos);
            demo.gpa.free(demo.vel);
        }

        fn reset(demo: *Demo, random: std.Random) !void {
            bodysystem.randomizeBodies(
                T,
                random,
                demo.pos,
                demo.vel,
                active_params.cluster_scale,
                active_params.velocity_scale,
                demo.num_bodies,
            );
            try demo.nbody_cuda.setPositions(demo.pos);
            try demo.nbody_cuda.setVelocities(demo.vel);
        }

        fn compareResults(demo: *Demo) !bool {
            var passed = true;

            try demo.nbody_cuda.update(0.001);

            var nbody_cpu: BodySystemCpu(T) = try .init(demo.gpa, demo.num_bodies);
            defer nbody_cpu.deinit(demo.gpa);

            @memcpy(nbody_cpu.pos, demo.pos);
            @memcpy(nbody_cpu.vel, demo.vel);

            nbody_cpu.update(0.001);

            const cuda_pos = try demo.gpa.alloc(T, demo.num_bodies * 4);
            defer demo.gpa.free(cuda_pos);
            try demo.nbody_cuda.getPositions(cuda_pos);
            const cpu_pos = nbody_cpu.pos;

            const tolerance: T = 0.0005;

            for (cpu_pos[0..demo.num_bodies], cuda_pos[0..demo.num_bodies]) |cpu, gpu| {
                if (@abs(cpu - gpu) > tolerance) {
                    passed = false;
                    std.log.err("Error: (host){d:.6} != (device){d:.6}", .{ cpu, gpu });
                }
            }
            if (passed) std.log.info("  OK", .{});
            return passed;
        }

        fn runBenchmark(demo: *Demo, iterations: u32) !void {
            const start = try cuda.Event.create();
            defer start.destroy();
            const stop = try cuda.Event.create();
            defer stop.destroy();

            // Once without timing to prime the device.
            try demo.nbody_cuda.update(active_params.timestep);

            try start.record(.default);

            for (0..iterations) |_| try demo.nbody_cuda.update(active_params.timestep);

            try stop.record(.default);
            try stop.synchronize();
            const milliseconds = try start.elapsed(stop);

            const stats = computePerfStats(T, demo.num_bodies, milliseconds, iterations);

            std.log.info("{d} bodies, total time for {d} iterations: {d:.3} ms", .{ demo.num_bodies, iterations, milliseconds });
            std.log.info("= {d:.3} billion interactions per second", .{stats.interactions_per_second});
            std.log.info("= {d:.3} {s}-precision GFLOP/s at {d} flops per interaction", .{
                stats.gflops,
                if (T == f64) "double" else "single",
                flopsPerInteraction(T),
            });
        }
    };
}

fn run(comptime T: type, gpa: std.mem.Allocator, module: cuda.Module, num_bodies: u32, block_size: u32, benchmark: bool, num_iterations: u32) !bool {
    var demo: NBodyDemo(T) = try .init(gpa, module, num_bodies, block_size);
    defer demo.deinit();
    var prng: std.Random.DefaultPrng = .init(0);
    try demo.reset(prng.random());

    if (benchmark) {
        try demo.runBenchmark(num_iterations);
        return true;
    }
    return try demo.compareResults();
}

fn showHelp() void {
    std.log.info("-fp64             (use double precision floating point values for simulation)", .{});
    std.log.info("-benchmark        (run benchmark to measure performance)", .{});
    std.log.info("-numbodies=<N>    (number of bodies (>= 1) to run in simulation)", .{});
    std.log.info("-device=<d>       (where d=0,1,2.... for the CUDA device to use)", .{});
    std.log.info("-compare          (compares simulation results running once on the default GPU and once on the CPU)", .{});
}

/// Returns the value of the command line argument `-<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) !?u32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') return try std.fmt.parseInt(u32, flag[name.len + 1 ..], 10);
    }
    return null;
}

fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len) continue;
        if (std.mem.eql(u8, flag, name) or (std.mem.startsWith(u8, flag, name) and flag.len > name.len and flag[name.len] == '=')) return true;
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (hasFlag(args, "help")) {
        std.log.info("> Command line options", .{});
        showHelp();
        return;
    }

    std.log.info("Run \"nbody -benchmark [-numbodies=<numBodies>]\" to measure performance.", .{});
    showHelp();

    std.log.info("NOTE: The CUDA Samples are not meant for performance measurements. " ++
        "Results may vary when GPU Boost is enabled.", .{});

    const benchmark = hasFlag(args, "benchmark");
    const fp64 = hasFlag(args, "fp64");

    std.log.info("> Simulation data stored in video memory", .{});
    std.log.info("> {s} precision floating point simulation", .{if (fp64) "Double" else "Single"});
    std.log.info("> 1 Devices used for simulation", .{});

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();

    const capability = try device.computeCapability();
    var name_buffer: [100]u8 = undefined;
    std.log.info("> Compute {d}.{d} CUDA device: [{s}]", .{ capability.major, capability.minor, try device.name(&name_buffer) });

    const num_iterations = if (try flagValue(args, "i")) |value| if (value > 0) value else 10 else 10;
    const block_size = if (try flagValue(args, "blockSize")) |value| if (value > 0) value else 256 else 256;

    var num_bodies: u32 = if (benchmark)
        block_size * 4 * @as(u32, @intCast(try device.attribute(.multiprocessor_count)))
    else
        4096;

    if (try flagValue(args, "numbodies")) |value| {
        num_bodies = value;
        if (num_bodies < 1) {
            std.log.err("Error: \"number of bodies\" specified {d} is invalid.  Value should be >= 1", .{num_bodies});
            return;
        } else if (num_bodies % block_size != 0) {
            const new_num_bodies = ((num_bodies / block_size) + 1) * block_size;
            std.log.warn("Warning: \"number of bodies\" specified {d} is not a multiple of {d}.", .{ num_bodies, block_size });
            std.log.info("Rounding up to the nearest multiple: {d}.", .{new_num_bodies});
            num_bodies = new_num_bodies;
        } else {
            std.log.info("number of bodies = {d}", .{num_bodies});
        }
    }

    const scales: ?struct { f32, f32 } = switch (num_bodies) {
        0...1024 => .{ 1.52, 2 },
        1025...2048 => .{ 1.56, 2.64 },
        2049...4096 => .{ 1.68, 2.98 },
        4097...8192 => .{ 1.98, 2.9 },
        8193...16384 => .{ 1.54, 8 },
        16385...32768 => .{ 1.44, 11 },
        else => null,
    };
    if (scales) |s| {
        active_params.cluster_scale, active_params.velocity_scale = s;
    }

    const module = try cuda.Module.load(@embedFile("bodysystemcuda_kernel.ptx"));
    defer module.unload();

    const passed = if (fp64)
        try run(f64, init.gpa, module, num_bodies, block_size, benchmark, num_iterations)
    else
        try run(f32, init.gpa, module, num_bodies, block_size, benchmark, num_iterations);
    if (!passed) std.process.exit(1);
}
