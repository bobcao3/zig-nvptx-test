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

//! Transpose of a floating-point matrix. The program runs several transpose kernels and reports
//! their effective bandwidth. The whitepaper in the docs folder of the CUDA samples project
//! "transpose" describes the performance study in detail.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("transpose_common.zig");

const tile_dim = common.tile_dim;
const block_rows = common.block_rows;

const sample_name = "Transpose";

/// The number of repetitions used for timing.
const num_reps = 100;

fn floor(a: u32, b: u32) u32 {
    return a - a % b;
}

const Kernel = enum {
    copy,
    copy_shared_mem,
    transpose_naive,
    transpose_coalesced,
    transpose_no_bank_conflicts,
    transpose_coarse_grained,
    transpose_fine_grained,
    transpose_diagonal,

    fn functionName(kernel: Kernel) [:0]const u8 {
        return switch (kernel) {
            .copy => "copy",
            .copy_shared_mem => "copySharedMem",
            .transpose_naive => "transposeNaive",
            .transpose_coalesced => "transposeCoalesced",
            .transpose_no_bank_conflicts => "transposeNoBankConflicts",
            .transpose_coarse_grained => "transposeCoarseGrained",
            .transpose_fine_grained => "transposeFineGrained",
            .transpose_diagonal => "transposeDiagonal",
        };
    }

    fn description(kernel: Kernel) []const u8 {
        return switch (kernel) {
            .copy => "simple copy       ",
            .copy_shared_mem => "shared memory copy",
            .transpose_naive => "naive             ",
            .transpose_coalesced => "coalesced         ",
            .transpose_no_bank_conflicts => "optimized         ",
            .transpose_coarse_grained => "coarse-grained    ",
            .transpose_fine_grained => "fine-grained      ",
            .transpose_diagonal => "diagonal          ",
        };
    }
};

fn computeTransposeGold(gold: []f32, idata: []const f32, size_x: u32, size_y: u32) void {
    for (0..size_y) |y| {
        for (0..size_x) |x| {
            gold[x * size_y + y] = idata[y * size_x + x];
        }
    }
}

/// Returns whether every element of `data` differs from `reference` by at most `epsilon`.
fn compareData(reference: []const f32, data: []const f32, epsilon: f32) bool {
    var result = true;
    for (reference, data) |expected, actual| {
        const diff = expected - actual;
        result = result and diff <= epsilon and diff >= -epsilon;
    }
    return result;
}

/// Returns the value of the command line argument `-<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) !?u32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') {
            return try std.fmt.parseInt(u32, flag[name.len + 1 ..], 10);
        }
    }
    return null;
}

fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, std.mem.trimStart(u8, arg, "-"), name)) return true;
    }
    return false;
}

/// Returns the matrix size along one dimension from the flag `-<name>=<size>`, or the largest
/// recommended size.
fn matrixSize(
    args: []const [:0]const u8,
    name: []const u8,
    label: []const u8,
    max_tile_dim: u32,
) !u32 {
    const size = try flagValue(args, name) orelse return floor(max_tile_dim, 512);
    if (size > max_tile_dim) {
        std.log.warn("> MatrixSize {s} = {d} is greater than the recommended size = {d}", .{
            label,
            size,
            max_tile_dim,
        });
    } else {
        std.log.info("> MatrixSize {s} = {d}", .{ label, size });
    }
    return size;
}

fn showHelp() void {
    std.log.info("{s} : Command line options", .{sample_name});
    std.log.info("\t-device=n          (where n=0,1,2.... for the GPU device)", .{});
    std.log.info("> The default matrix size can be overridden with these parameters", .{});
    std.log.info("\t-dimX=row_dim_size (matrix row    dimensions)", .{});
    std.log.info("\t-dimY=col_dim_size (matrix column dimensions)", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("{s} Starting...", .{sample_name});
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (hasFlag(args, "help")) {
        showHelp();
        return;
    }

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();
    var name_buffer: [256]u8 = undefined;
    const device_name = try device.name(&name_buffer);
    const capability = try device.computeCapability();
    const multiprocessor_count = try device.attribute(.multiprocessor_count);
    const cores = cuda.coresPerMultiprocessor(capability.major, capability.minor);

    // The scaling factor for GPUs with fewer multiprocessors.
    const scale_factor = @max(192.0 / @as(f32, @floatFromInt(cores * multiprocessor_count)), 1.0);

    std.log.info("> Device {d}: \"{s}\"", .{ device.ordinal, device_name });
    std.log.info("> SM Capability {d}.{d} detected:", .{ capability.major, capability.minor });

    // The number of tiles of the matrix transpose performance tests. The size is rounded down
    // to a multiple of 512.
    const matrix_size_test = 512;
    const default_matrix_size = 1024;
    const floored_matrix_size = floor(default_matrix_size, 512);
    const max_tiles: f32 = @floatFromInt(floored_matrix_size * floored_matrix_size / (tile_dim * tile_dim));
    const total_tiles = max_tiles / scale_factor;
    const tiles_per_side: u32 = @intFromFloat(@floor(@sqrt(total_tiles)));
    var max_matrix_dim = floor(tiles_per_side * tile_dim, matrix_size_test);
    if (max_matrix_dim == 0) max_matrix_dim = matrix_size_test;

    std.log.info("> [{s}] has {d} MP(s) x {d} (Cores/MP) = {d} (Cores)", .{
        device_name,
        multiprocessor_count,
        cores,
        cores * multiprocessor_count,
    });
    std.log.info("> Compute performance scaling factor = {d:.2}", .{scale_factor});

    const size_x = try matrixSize(args, "dimX", "X", max_matrix_dim);
    const size_y = try matrixSize(args, "dimY", "Y", max_matrix_dim);

    if (size_x != size_y) {
        std.log.err("[{s}] does not support non-square matrices (row_dim_size({d}) != " ++
            "col_dim_size({d}))", .{ sample_name, size_x, size_y });
        std.log.err("Exiting...", .{});
        std.process.exit(1);
    }

    if (size_x % tile_dim != 0 or size_y % tile_dim != 0) {
        std.log.err("[{s}] Matrix size must be integral multiple of tile " ++
            "size", .{sample_name});
        std.log.err("Exiting...", .{});
        std.process.exit(1);
    }

    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = size_x / tile_dim, .y = size_y / tile_dim },
        .block = .{ .x = tile_dim, .y = block_rows },
    };
    if (config.grid.x < 1 or config.grid.y < 1) {
        std.log.err("[{s}] grid size computation incorrect in test", .{sample_name});
        std.log.err("Exiting...", .{});
        std.process.exit(1);
    }

    const module = try cuda.Module.load(@embedFile("transpose_kernel.ptx"));
    defer module.unload();

    const element_count = size_x * size_y;
    const mem_size: f32 = @floatFromInt(@sizeOf(f32) * @as(usize, element_count));

    const h_idata = try gpa.alloc(f32, element_count);
    defer gpa.free(h_idata);
    const h_odata = try gpa.alloc(f32, element_count);
    defer gpa.free(h_odata);
    const transpose_gold = try gpa.alloc(f32, element_count);
    defer gpa.free(transpose_gold);

    const d_idata = try cuda.alloc(f32, element_count);
    defer d_idata.free();
    const d_odata = try cuda.alloc(f32, element_count);
    defer d_odata.free();

    for (h_idata, 0..) |*element, i| element.* = @floatFromInt(i);
    try d_idata.copyFrom(h_idata);

    computeTransposeGold(transpose_gold, h_idata, size_x, size_y);

    std.log.info("Matrix size: {d}x{d} ({d}x{d} tiles), tile size: {d}x{d}, " ++
        "block size: {d}x{d}", .{
        size_x,
        size_y,
        size_x / tile_dim,
        size_y / tile_dim,
        tile_dim,
        tile_dim,
        tile_dim,
        block_rows,
    });

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    var success = true;

    for (std.enums.values(Kernel)) |kernel_kind| {
        const kernel = try module.function(kernel_kind.functionName());
        const args_list = .{ d_odata, d_idata, size_x, size_y };

        // The fine-grained and the coarse-grained kernels are not full transposes, so the
        // comparison bypasses them.
        const gold = switch (kernel_kind) {
            .copy, .copy_shared_mem => h_idata,
            .transpose_coarse_grained, .transpose_fine_grained => h_odata,
            else => transpose_gold,
        };

        // A warmup launch avoids timing the startup.
        try kernel.launch(config, args_list);

        try start.record(.default);
        for (0..num_reps) |_| try kernel.launch(config, args_list);
        try stop.record(.default);
        try stop.synchronize();
        const kernel_time = try start.elapsed(stop);

        try d_odata.copyTo(h_odata);
        if (!compareData(gold, h_odata, 0.01)) {
            std.log.err("*** {s} kernel FAILED ***", .{kernel_kind.description()});
            success = false;
        }

        const kernel_time_per_rep = kernel_time / num_reps;
        const kernel_bandwidth = 2.0 * 1000.0 * mem_size / (1024 * 1024 * 1024) / kernel_time_per_rep;
        std.log.info("transpose {s}, Throughput = {d:.4} GB/s, Time = {d:.5} ms, Size = {d} fp32 " ++
            "elements, NumDevsUsed = {d}, Workgroup = {d}", .{
            kernel_kind.description(),
            kernel_bandwidth,
            kernel_time_per_rep,
            element_count,
            1,
            tile_dim * block_rows,
        });

        // Resets d_odata to zero so that a kernel that does no work cannot pass with the
        // result of the previous kernel.
        @memset(h_odata, 0);
        try d_odata.copyFrom(h_odata);
    }

    if (!success) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }

    std.log.info("Test passed", .{});
}
