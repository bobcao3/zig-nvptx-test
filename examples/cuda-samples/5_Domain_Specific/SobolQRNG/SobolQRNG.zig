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
//
// Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights
// reserved.
// Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights
// reserved.

//! Sobol Quasi-random Number Generator example
//!
//! Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
//! http://people.maths.ox.ac.uk/~gilesm/
//!
//! and C code developed by Stephen Joe, University of Waikato, New Zealand
//! and Frances Kuo, University of New South Wales, Australia
//! http://web.maths.unsw.edu.au/~fkuo/sobol/
//!
//! For theoretical background see:
//!
//! P. Bratley and B.L. Fox.
//! Implementing Sobol's quasirandom sequence generator
//! http://portal.acm.org/citation.cfm?id=42288
//! ACM Trans. on Math. Software, 14(1):88-100, 1988
//!
//! S. Joe and F. Kuo.
//! Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
//! http://portal.acm.org/citation.cfm?id=641879
//! ACM Trans. on Math. Software, 29(1):49-57, 2003

const std = @import("std");
const cuda = @import("cuda");
const sobol = @import("sobol.zig");
const sobol_gold = @import("sobol_gold.zig");
const sobol_primitives = @import("sobol_primitives.zig").sobol_primitives;

const n_directions = sobol.n_directions;

const l1_error_tolerance = 1e-6;

const sample_name = "Sobol Quasi-Random Number Generator";

fn printHelp() void {
    std.log.info("Usage: <program name> <options>", .{});
    std.log.info("\t--vectors=M     specify number of vectors    (required)", .{});
    std.log.info("\t                The generator will output M vectors", .{});
    std.log.info("\t--dimensions=N  specify number of dimensions (required)", .{});
    std.log.info("\t                Each vector will consist of N components", .{});
}

/// Returns the value of the command line argument `--<name>=<value>`.
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

fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (std.mem.eql(u8, flag, name)) return true;
    }
    return false;
}

/// Launches the generator. All the draws of one dimension are in a contiguous region of memory,
/// followed by the next dimension and so on.
fn sobolGpu(
    sobol_gpu_kernel: cuda.Function,
    device: cuda.Device,
    n_vectors: u32,
    n_dimensions: u32,
    d_directions: cuda.Buffer(u32),
    d_output: cuda.Buffer(f32),
) !void {
    const threads_per_block = 64;
    const multiprocessor_count: u32 = @intCast(try device.attribute(.multiprocessor_count));

    // All work items of a work group process different vectors of the same dimension. The total
    // number of work groups is therefore a multiple of the number of dimensions.
    var grid: cuda.Dim3 = .{ .y = n_dimensions };

    // If the number of dimensions is large, the number of work groups equals the number of
    // dimensions. If the number of dimensions is small, for example less than four per
    // multiprocessor, the vectors are partitioned across work groups as well as work items.
    grid.x = if (n_dimensions < 4 * multiprocessor_count) 4 * multiprocessor_count else 1;

    // Caps the first grid dimension if the number of vectors is small.
    if (grid.x > n_vectors / threads_per_block) {
        grid.x = (n_vectors + threads_per_block - 1) / threads_per_block;
    }

    // Rounds up to a power of two, so that the stride is a power of two as the algorithm
    // requires.
    const target_grid_x = grid.x;
    grid.x = 1;
    while (grid.x < target_grid_x) grid.x *= 2;

    try sobol_gpu_kernel.launch(
        .{ .grid = grid, .block = .{ .x = threads_per_block } },
        .{ n_vectors, n_dimensions, d_directions, d_output },
    );
}

fn printGsamplesPerSecond(n_vectors: u32, n_dimensions: u32, time_ms: f64) void {
    if (time_ms < 1e-6) {
        std.log.warn("Gsamples/s: problem size too small to measure, try " ++
            "increasing number of vectors or dimensions", .{});
    } else {
        const samples = @as(f64, @floatFromInt(n_vectors)) * @as(f64, @floatFromInt(n_dimensions));
        std.log.info("Gsamples/s: {d}", .{samples * 1e-9 / (time_ms * 1e-3)});
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var ok = true;

    // The program generates n_vectors vectors of n_dimensions numbers.
    var n_vectors: i32 = 100000;
    var n_dimensions: i32 = 100;

    std.log.info("{s} Starting...", .{sample_name});

    if (hasFlag(args, "help")) {
        printHelp();
        return;
    }

    if (!hasFlag(args, "qatest")) {
        if (try flagValue(args, "vectors")) |value| {
            n_vectors = value;
            if (n_vectors < 1) {
                std.log.err("Illegal argument: number of vectors must be positive " ++
                    "(--vectors=N)", .{});
                ok = false;
            }
        }
        std.log.info("> number of vectors = {d}", .{n_vectors});

        if (try flagValue(args, "dimensions")) |value| {
            n_dimensions = value;
            if (n_dimensions < 1) {
                std.log.err("Illegal argument: number of dimensions must be positive " ++
                    "(--dimensions=N)", .{});
                ok = false;
            }
        }
        std.log.info("> number of dimensions = {d}", .{n_dimensions});
    }

    if (!ok) std.process.exit(1);

    const vectors: u32 = @intCast(n_vectors);
    const dimensions: u32 = @intCast(n_dimensions);
    if (dimensions > sobol_primitives.len) {
        std.log.err("The table of primitive polynomials has only {d} dimensions", .{
            sobol_primitives.len,
        });
        std.process.exit(1);
    }

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("sobol_gpu_kernel.ptx"));
    defer module.unload();
    const sobol_gpu_kernel = try module.function("sobolGpuKernel");

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    std.log.info("Allocating CPU memory...", .{});
    const h_directions = gpa.alloc(u32, dimensions * n_directions) catch |err|
        unableToAllocate("CPU", err);
    defer gpa.free(h_directions);
    const h_output_cpu = gpa.alloc(f32, vectors * dimensions) catch |err|
        unableToAllocate("CPU", err);
    defer gpa.free(h_output_cpu);
    const h_output_gpu = gpa.alloc(f32, vectors * dimensions) catch |err|
        unableToAllocate("CPU", err);
    defer gpa.free(h_output_gpu);

    std.log.info("Allocating GPU memory...", .{});
    const d_directions = cuda.alloc(u32, dimensions * n_directions) catch |err|
        unableToAllocate("GPU", err);
    defer d_directions.free();
    const d_output = cuda.alloc(f32, vectors * dimensions) catch |err|
        unableToAllocate("GPU", err);
    defer d_output.free();

    // Initializes the direction numbers on the host.
    std.log.info("Initializing direction numbers...", .{});
    sobol_gold.initSobolDirectionVectors(dimensions, h_directions);

    // Copies the direction numbers to the device.
    std.log.info("Copying direction numbers to device...", .{});
    try d_directions.copyFrom(h_directions);
    try cuda.synchronize();

    // Executes the generator on the device.
    std.log.info("Executing QRNG on GPU...", .{});
    try start.record(.default);
    try sobolGpu(sobol_gpu_kernel, device, vectors, dimensions, d_directions, d_output);
    try stop.record(.default);
    try stop.synchronize();
    printGsamplesPerSecond(vectors, dimensions, try start.elapsed(stop));

    std.log.info("Reading results from GPU...", .{});
    try d_output.copyTo(h_output_gpu);

    // Executes the generator on the host.
    std.log.info("Executing QRNG on CPU...", .{});
    const cpu_start = std.Io.Clock.awake.now(init.io);
    sobol_gold.sobolCpu(vectors, dimensions, h_directions, h_output_cpu);
    const cpu_time = cpu_start.durationTo(std.Io.Clock.awake.now(init.io));
    const cpu_time_ms = @as(f64, @floatFromInt(cpu_time.toNanoseconds())) * 1e-6;
    printGsamplesPerSecond(vectors, dimensions, cpu_time_ms);

    // Checks the results.
    std.log.info("Checking results...", .{});
    var l1norm_diff: f32 = 0.0;
    var l1norm_ref: f32 = 0.0;
    var l1error: f32 = undefined;

    if (vectors == 1) {
        // The vector is exactly 0 if n_vectors is 1.
        for (0..dimensions) |d| {
            const reference = h_output_cpu[d * vectors];
            l1norm_diff += @abs(h_output_gpu[d * vectors] - reference);
            l1norm_ref += @abs(reference);
        }

        l1error = l1norm_diff;

        if (l1norm_ref != 0) {
            std.log.err("Error: L1-Norm of the reference is not zero (for single " ++
                "vector), golden generator appears broken", .{});
        } else {
            std.log.info("L1-Error: {d}", .{l1error});
        }
    } else {
        for (0..dimensions) |d| {
            for (0..vectors) |v| {
                const reference = h_output_cpu[d * vectors + v];
                l1norm_diff += @abs(h_output_gpu[d * vectors + v] - reference);
                l1norm_ref += @abs(reference);
            }
        }

        l1error = l1norm_diff / l1norm_ref;

        if (l1norm_ref == 0) {
            std.log.err("Error: L1-Norm of the reference is zero, golden generator " ++
                "appears broken", .{});
        } else {
            std.log.info("L1-Error: {d}", .{l1error});
        }
    }

    std.log.info("Shutting down...", .{});

    if (!(l1error < l1_error_tolerance)) std.process.exit(1);
}

fn unableToAllocate(comptime where: []const u8, err: anyerror) noreturn {
    std.log.err("Caught exception: {t}", .{err});
    std.log.err("Unable to allocate " ++ where ++ " memory (try running with fewer " ++
        "vectors/dimensions)", .{});
    std.process.exit(1);
}
