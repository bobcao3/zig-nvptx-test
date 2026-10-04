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

//! This sample implements 64-bin and 256-bin histogram calculation of an arbitrary-sized 8-bit
//! data array.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("histogram_common.zig");

const num_runs = 16;

fn histogram64Cpu(h_histogram: *[common.histogram64_bin_count]u32, h_data: []const u32) void {
    @memset(h_histogram, 0);
    for (h_data) |data| {
        inline for (.{ 2, 10, 18, 26 }) |shift| h_histogram[(data >> shift) & 0x3F] += 1;
    }
}

fn histogram256Cpu(h_histogram: *[common.histogram256_bin_count]u32, h_data: []const u32) void {
    @memset(h_histogram, 0);
    for (h_data) |data| {
        inline for (.{ 0, 8, 16, 24 }) |shift| h_histogram[(data >> shift) & 0xFF] += 1;
    }
}

/// Rounds a / b up to the next integer.
fn divUp(a: u32, b: u32) u32 {
    return std.math.divCeil(u32, a, b) catch unreachable;
}

/// Snaps a down to a multiple of b.
fn snapDown(a: u32, b: u32) u32 {
    return a - a % b;
}

const Histogram64 = struct {
    histogram: cuda.Function,
    merge: cuda.Function,
    d_partial_histograms: cuda.Buffer(u32),

    // The partial histograms of up to 480 MB of input data fit into the buffer.
    const max_partial_histogram_count = 32768;
    const data_size = @sizeOf(@Vector(4, u32));

    fn init(module: cuda.Module) !Histogram64 {
        comptime std.debug.assert(common.histogram64_work_group_size % (4 * common.shared_memory_banks) == 0);
        return .{
            .histogram = try module.function("histogram64Kernel"),
            .merge = try module.function("mergeHistogram64Kernel"),
            .d_partial_histograms = try cuda.alloc(
                u32,
                max_partial_histogram_count * common.histogram64_bin_count,
            ),
        };
    }

    fn deinit(self: Histogram64) void {
        self.d_partial_histograms.free();
    }

    fn run(self: Histogram64, d_histogram: cuda.Buffer(u32), d_data: cuda.Buffer(u8), byte_count: u32) !void {
        const histogram_count = divUp(
            byte_count,
            common.histogram64_work_group_size * snapDown(255, data_size),
        );
        std.debug.assert(byte_count % data_size == 0);
        std.debug.assert(histogram_count <= max_partial_histogram_count);

        try self.histogram.launch(
            .{ .grid = .{ .x = histogram_count }, .block = .{ .x = common.histogram64_work_group_size } },
            .{ self.d_partial_histograms, d_data, byte_count / data_size },
        );
        try self.merge.launch(
            .{ .grid = .{ .x = common.histogram64_bin_count }, .block = .{ .x = common.merge_work_group_size } },
            .{ d_histogram, self.d_partial_histograms, histogram_count },
        );
    }
};

const Histogram256 = struct {
    histogram: cuda.Function,
    merge: cuda.Function,
    d_partial_histograms: cuda.Buffer(u32),

    const partial_histogram_count = 240;

    fn init(module: cuda.Module) !Histogram256 {
        return .{
            .histogram = try module.function("histogram256Kernel"),
            .merge = try module.function("mergeHistogram256Kernel"),
            .d_partial_histograms = try cuda.alloc(
                u32,
                partial_histogram_count * common.histogram256_bin_count,
            ),
        };
    }

    fn deinit(self: Histogram256) void {
        self.d_partial_histograms.free();
    }

    fn run(self: Histogram256, d_histogram: cuda.Buffer(u32), d_data: cuda.Buffer(u8), byte_count: u32) !void {
        std.debug.assert(byte_count % @sizeOf(u32) == 0);
        try self.histogram.launch(
            .{ .grid = .{ .x = partial_histogram_count }, .block = .{ .x = common.histogram256_work_group_size } },
            .{ self.d_partial_histograms, d_data, byte_count / @sizeOf(u32) },
        );
        try self.merge.launch(
            .{ .grid = .{ .x = common.histogram256_bin_count }, .block = .{ .x = common.merge_work_group_size } },
            .{ d_histogram, self.d_partial_histograms, @as(u32, partial_histogram_count) },
        );
    }
};

/// Runs `histogram` in `num_runs` timed iterations after one warmup iteration and prints the
/// throughput. `Histogram` is `Histogram64` or `Histogram256`.
fn timeRuns(
    histogram: anytype,
    name: []const u8,
    work_group_size: u32,
    d_histogram: cuda.Buffer(u32),
    d_data: cuda.Buffer(u8),
    byte_count: u32,
) !void {
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    try histogram.run(d_histogram, d_data, byte_count);
    try cuda.synchronize();
    try start.record(.default);
    for (0..num_runs) |_| try histogram.run(d_histogram, d_data, byte_count);
    try stop.record(.default);
    try stop.synchronize();

    const avg_secs = 1.0e-3 * try start.elapsed(stop) / num_runs;
    const megabytes = @as(f64, @floatFromInt(byte_count)) * 1.0e-6;
    std.log.info("{s}() time (average) : {d:.5} sec, {d:.4} MB/sec", .{ name, avg_secs, megabytes / avg_secs });
    std.log.info("{s}, Throughput = {d:.4} MB/s, Time = {d:.5} s, Size = {d} Bytes, " ++
        "NumDevsUsed = {d}, Workgroup = {d}", .{ name, megabytes / avg_secs, avg_secs, byte_count, 1, work_group_size });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var byte_count: u32 = 64 * 1048576;

    std.log.info("[histogram] - Starting...", .{});

    const device = try cuda.init(0);
    defer device.deinit();
    var name_buffer: [256]u8 = undefined;
    const capability = try device.computeCapability();
    std.log.info("CUDA device [{s}] has {d} Multi-Processors, Compute {d}.{d}", .{
        try device.name(&name_buffer),
        try device.attribute(.multiprocessor_count),
        capability.major,
        capability.minor,
    });

    // Optional command line multiplier of the size of the array to histogram.
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (std.mem.startsWith(u8, flag, "sizemult=")) {
            const size_mult = try std.fmt.parseInt(u32, flag["sizemult=".len..], 10);
            byte_count *= std.math.clamp(size_mult, 1, 10);
        }
    }

    std.log.info("Initializing data...", .{});
    std.log.info("...allocating CPU memory.", .{});
    const h_data = try gpa.alignedAlloc(u8, .of(u32), byte_count);
    defer gpa.free(h_data);
    var h_histogram_cpu: [common.histogram256_bin_count]u32 = undefined;
    var h_histogram_gpu: [common.histogram256_bin_count]u32 = undefined;

    std.log.info("...generating input data", .{});
    var prng: std.Random.DefaultPrng = .init(2009);
    prng.random().bytes(h_data);

    std.log.info("...allocating GPU memory and copying input data", .{});
    const d_data = try cuda.alloc(u8, byte_count);
    defer d_data.free();
    const d_histogram = try cuda.alloc(u32, common.histogram256_bin_count);
    defer d_histogram.free();
    try d_data.copyFrom(h_data);

    const words: []const u32 = @ptrCast(h_data);
    var pass = true;

    {
        std.log.info("Starting up 64-bin histogram...", .{});
        const module = try cuda.Module.load(@embedFile("histogram64_kernel.ptx"));
        defer module.unload();
        const histogram64: Histogram64 = try .init(module);
        defer {
            std.log.info("Shutting down 64-bin histogram...", .{});
            histogram64.deinit();
        }

        std.log.info("Running 64-bin GPU histogram for {d} bytes ({d} runs)...", .{ byte_count, num_runs });
        try timeRuns(histogram64, "histogram64", common.histogram64_work_group_size, d_histogram, d_data, byte_count);

        std.log.info("Validating GPU results...", .{});
        std.log.info(" ...reading back GPU results", .{});
        try d_histogram.copyTo(&h_histogram_gpu);

        std.log.info(" ...histogram64CPU()", .{});
        histogram64Cpu(h_histogram_cpu[0..common.histogram64_bin_count], words);

        std.log.info(" ...comparing the results...", .{});
        const bins = common.histogram64_bin_count;
        if (!std.mem.eql(u32, h_histogram_gpu[0..bins], h_histogram_cpu[0..bins])) pass = false;
        if (pass) std.log.info(" ...64-bin histograms match", .{}) else std.log.err(" ***64-bin histograms do not match!!!***", .{});
    }

    {
        std.log.info("Initializing 256-bin histogram...", .{});
        const module = try cuda.Module.load(@embedFile("histogram256_kernel.ptx"));
        defer module.unload();
        const histogram256: Histogram256 = try .init(module);
        defer {
            std.log.info("Shutting down 256-bin histogram...", .{});
            histogram256.deinit();
        }

        std.log.info("Running 256-bin GPU histogram for {d} bytes ({d} runs)...", .{ byte_count, num_runs });
        try timeRuns(histogram256, "histogram256", common.histogram256_work_group_size, d_histogram, d_data, byte_count);

        std.log.info("Validating GPU results...", .{});
        std.log.info(" ...reading back GPU results", .{});
        try d_histogram.copyTo(&h_histogram_gpu);

        std.log.info(" ...histogram256CPU()", .{});
        histogram256Cpu(&h_histogram_cpu, words);

        std.log.info(" ...comparing the results", .{});
        if (!std.mem.eql(u32, &h_histogram_gpu, &h_histogram_cpu)) pass = false;
        if (pass) std.log.info(" ...256-bin histograms match", .{}) else std.log.err(" ***256-bin histograms do not match!!!***", .{});
    }

    std.log.info("Shutting down...", .{});
    std.log.info("NOTE: The CUDA Samples are not meant for performance measurements. " ++
        "Results may vary when GPU Boost is enabled.", .{});
    std.log.info("[histogram] - Test Summary", .{});

    if (!pass) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }
    std.log.info("Test passed", .{});
}
