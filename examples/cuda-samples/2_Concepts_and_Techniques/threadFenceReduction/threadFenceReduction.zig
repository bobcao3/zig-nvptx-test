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

//! Parallel reduction.
//!
//! This sample shows how to perform a reduction operation on an array of values to produce a
//! single value in a single kernel (as opposed to two or more kernel calls as shown in the
//! "reduction" CUDA Sample). Single-pass reduction requires global atomic instructions and a
//! memory fence.
//!
//! Reductions are a very common computation in parallel algorithms. Any time an array of values
//! needs to be reduced to a single value using a binary associative operator, a reduction can be
//! used. Example applications include statistics computations such as mean and standard
//! deviation, and image processing applications such as finding the total luminance of an image.
//!
//! This code performs sum reductions, but any associative operator such as min() or max() could
//! also be used.
//!
//! It assumes the input size is a power of 2.
//!
//! Command line arguments:
//!
//! - `--shmoo`: Test performance for 1 to 32M elements with each of the 7 different kernels.
//! - `--n=<N>`: Specify the number of elements to reduce (default 1048576).
//! - `--threads=<N>`: Specify the number of threads per block (default 128).
//! - `--maxblocks=<N>`: Specify the maximum number of thread blocks to launch (kernel 6 only,
//!   default 64).
//! - `--cpufinal`: Read back the per-block results and do the final sum of block sums on the CPU
//!   (default false).
//! - `--cputhresh=<N>`: The threshold of the number of block sums below which to perform a CPU
//!   final reduction (default 1).
//! - `--multipass`: Use a multipass reduction instead of a single-pass reduction.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("threadFenceReduction_common.zig");

const sdk_sample = "threadFenceReduction";

fn reduceCpu(data: []const f32) f32 {
    var sum = data[0];
    var c: f32 = 0.0;

    for (data[1..]) |element| {
        const y = element - c;
        const t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

fn isPow2(x: u32) bool {
    return x & (x -% 1) == 0;
}

const Blocks = struct { blocks: u32, threads: u32 };

/// Computes the number of threads and blocks to use for the reduction. The number of threads per
/// block is the minimum of `max_threads` and n/2.
fn getNumBlocksAndThreads(n: u32, max_blocks: u32, max_threads: u32) Blocks {
    var threads: u32 = undefined;
    var blocks: u32 = undefined;
    if (n == 1) {
        threads = 1;
        blocks = 1;
    } else {
        threads = if (n < max_threads * 2)
            std.math.ceilPowerOfTwoAssert(u32, n / 2)
        else
            max_threads;
        blocks = @max(1, n / (threads * 2));
    }

    blocks = @min(max_blocks, blocks);
    return .{ .blocks = blocks, .threads = threads };
}

fn setRetirementCount(module: cuda.Module, ret_cnt: u32) !void {
    const retirement_count = try module.global(u32, "retirement_count");
    try retirement_count.copyFrom((&ret_cnt)[0..1]);
}

fn launchKernel(
    module: cuda.Module,
    comptime name: [:0]const u8,
    config: cuda.LaunchConfig,
    args: anytype,
) !void {
    const kernel = try module.function(name);
    try kernel.launch(config, args);
}

fn launchBlockSize(
    module: cuda.Module,
    comptime kernel: []const u8,
    size: u32,
    config: cuda.LaunchConfig,
    args: anytype,
) !void {
    switch (isPow2(size)) {
        inline else => |n_is_pow2| inline for (common.block_sizes) |block_size| {
            if (config.block.x == block_size) {
                const name = comptime common.kernelName(kernel, block_size, n_is_pow2);
                try launchKernel(module, name, config, args);
            }
        },
    }
}

fn reduce(
    module: cuda.Module,
    size: u32,
    threads: u32,
    blocks: u32,
    d_idata: cuda.Buffer(f32),
    d_odata: cuda.Buffer(f32),
) !void {
    const smem_size = if (threads <= 32) 2 * threads * @sizeOf(f32) else threads * @sizeOf(f32);
    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = blocks },
        .block = .{ .x = threads },
        .shared_memory_bytes = smem_size,
    };
    try launchBlockSize(module, "reduceMultiPass", size, config, .{ d_idata, d_odata, size });
}

fn reduceSinglePass(
    module: cuda.Module,
    size: u32,
    threads: u32,
    blocks: u32,
    d_idata: cuda.Buffer(f32),
    d_odata: cuda.Buffer(f32),
) !void {
    const config: cuda.LaunchConfig = .{
        .grid = .{ .x = blocks },
        .block = .{ .x = threads },
        .shared_memory_bytes = threads * @sizeOf(f32),
    };
    try launchBlockSize(module, "reduceSinglePass", size, config, .{ d_idata, d_odata, size });
}

const Timer = struct {
    io: std.Io,
    start_time: std.Io.Timestamp = undefined,
    total_seconds: f64 = 0,
    laps: u32 = 0,

    fn start(timer: *Timer) void {
        timer.start_time = std.Io.Clock.awake.now(timer.io);
    }

    fn stop(timer: *Timer) void {
        const elapsed = timer.start_time.durationTo(std.Io.Clock.awake.now(timer.io));
        timer.total_seconds += @as(f64, @floatFromInt(elapsed.toNanoseconds())) * 1.0e-9;
        timer.laps += 1;
    }

    fn reset(timer: *Timer) void {
        timer.total_seconds = 0;
        timer.laps = 0;
    }

    fn averageMilliseconds(timer: Timer) f64 {
        return 1000.0 * timer.total_seconds / @as(f64, @floatFromInt(timer.laps));
    }
};

const Settings = struct {
    max_threads: u32,
    max_blocks: u32,
    multi_pass: bool,
    cpu_final_reduction: bool,
    cpu_final_threshold: u32,
};

/// Performs the reduction of the input data multiple times and measures the average reduction
/// time.
fn benchmarkReduce(
    module: cuda.Module,
    n: u32,
    num_threads: u32,
    num_blocks: u32,
    settings: Settings,
    test_iterations: u32,
    timer: *Timer,
    h_odata: []f32,
    d_idata: cuda.Buffer(f32),
    d_odata: cuda.Buffer(f32),
) !f32 {
    var gpu_result: f32 = 0;
    var need_readback = true;

    for (0..test_iterations) |_| {
        gpu_result = 0;
        try setRetirementCount(module, 0);

        try cuda.synchronize();
        timer.start();

        if (settings.multi_pass) {
            try reduce(module, n, num_threads, num_blocks, d_idata, d_odata);

            if (settings.cpu_final_reduction) {
                try d_odata.copyTo(h_odata[0..num_blocks]);

                for (h_odata[0..num_blocks]) |partial_sum| gpu_result += partial_sum;

                need_readback = false;
            } else {
                var s = num_blocks;

                while (s > settings.cpu_final_threshold) {
                    const next = getNumBlocksAndThreads(
                        s,
                        settings.max_blocks,
                        settings.max_threads,
                    );

                    try reduce(module, s, next.threads, next.blocks, d_odata, d_odata);

                    s = s / (next.threads * 2);
                }

                if (s > 1) {
                    try d_odata.copyTo(h_odata[0..s]);

                    for (h_odata[0..s]) |partial_sum| gpu_result += partial_sum;

                    need_readback = false;
                }
            }
        } else {
            try reduceSinglePass(module, n, num_threads, num_blocks, d_idata, d_odata);
        }

        try cuda.synchronize();
        timer.stop();
    }

    if (need_readback) try d_odata.copyTo((&gpu_result)[0..1]);

    return gpu_result;
}

fn randomElement(random: std.Random) f32 {
    const rand_max: f32 = @floatFromInt(@as(i32, std.math.maxInt(i32)));
    return @as(f32, @floatFromInt(random.int(u8))) / rand_max;
}

/// Calls `benchmarkReduce` for a range of array sizes and prints a report in comma-separated
/// value format that can be used for generating a "shmoo" plot showing the performance for each
/// kernel variation over a wide range of input sizes.
fn shmoo(
    gpa: std.mem.Allocator,
    io: std.Io,
    module: cuda.Module,
    min_n: u32,
    max_n: u32,
    max_threads: u32,
    max_blocks: u32,
) !void {
    const h_idata = try gpa.alloc(f32, max_n);
    defer gpa.free(h_idata);

    var prng: std.Random.DefaultPrng = .init(1);
    for (h_idata) |*element| element.* = randomElement(prng.random());

    const max_num_blocks = @min(65535, max_n / max_threads);

    const h_odata = try gpa.alloc(f32, max_num_blocks);
    defer gpa.free(h_odata);

    const d_idata = try cuda.dupe(f32, h_idata);
    defer d_idata.free();
    const d_odata = try cuda.dupe(f32, h_idata[0..max_num_blocks]);
    defer d_odata.free();

    try reduce(module, max_n, max_threads, max_num_blocks, d_idata, d_odata);
    const test_iterations = 100;

    var timer: Timer = .{ .io = io };

    std.log.info("N, {d} blocks one pass, {d} blocks multipass", .{ max_blocks, max_blocks });

    var i = min_n;
    while (i <= max_n) : (i *= 2) {
        var row: std.Io.Writer.Allocating = .init(gpa);
        defer row.deinit();
        try row.writer.print("{d}", .{i});

        for ([_]bool{ false, true }) |multi_pass| {
            timer.reset();
            const blocks_and_threads = getNumBlocksAndThreads(i, max_blocks, max_threads);

            const settings: Settings = .{
                .max_threads = max_threads,
                .max_blocks = max_blocks,
                .multi_pass = multi_pass,
                .cpu_final_reduction = false,
                .cpu_final_threshold = 1,
            };
            _ = try benchmarkReduce(
                module,
                i,
                blocks_and_threads.threads,
                blocks_and_threads.blocks,
                settings,
                test_iterations,
                &timer,
                h_odata,
                d_idata,
                d_odata,
            );

            try row.writer.print(", {d:.6}", .{timer.averageMilliseconds()});
        }
        std.log.info("{s}", .{row.written()});
    }
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
        const flag = std.mem.trimStart(u8, arg, "-");
        if (std.mem.eql(u8, flag, name) or
            (std.mem.startsWith(u8, flag, name) and flag.len > name.len and flag[name.len] == '='))
        {
            return true;
        }
    }
    return false;
}

/// Runs the reduction test and returns whether the GPU result matches the CPU result.
fn runTest(init: std.process.Init, args: []const [:0]const u8, module: cuda.Module) !bool {
    var size: u32 = 1 << 20;
    var settings: Settings = .{
        .max_threads = 128,
        .max_blocks = 64,
        .multi_pass = false,
        .cpu_final_reduction = false,
        .cpu_final_threshold = 1,
    };

    if (try flagValue(args, "n")) |value| size = value;
    if (try flagValue(args, "threads")) |value| settings.max_threads = value;
    if (try flagValue(args, "maxblocks")) |value| settings.max_blocks = value;

    std.log.info("{d} elements", .{size});
    std.log.info("{d} threads (max)", .{settings.max_threads});

    settings.cpu_final_reduction = hasFlag(args, "cpufinal");
    settings.multi_pass = hasFlag(args, "multipass");

    if (try flagValue(args, "cputhresh")) |value| settings.cpu_final_threshold = value;

    if (hasFlag(args, "shmoo")) {
        const max_n = 33554432;
        try shmoo(init.gpa, init.io, module, 1, max_n, settings.max_threads, settings.max_blocks);
        return false;
    }

    const bytes = size * @sizeOf(f32);

    const h_idata = try init.gpa.alloc(f32, size);
    defer init.gpa.free(h_idata);

    var prng: std.Random.DefaultPrng = .init(1);
    for (h_idata) |*element| element.* = randomElement(prng.random());

    const blocks_and_threads = getNumBlocksAndThreads(
        size,
        settings.max_blocks,
        settings.max_threads,
    );
    const num_blocks = blocks_and_threads.blocks;
    const num_threads = blocks_and_threads.threads;

    if (num_blocks == 1) settings.cpu_final_threshold = 1;

    const h_odata = try init.gpa.alloc(f32, num_blocks);
    defer init.gpa.free(h_odata);

    std.log.info("{d} blocks", .{num_blocks});

    const d_idata = try cuda.dupe(f32, h_idata);
    defer d_idata.free();
    const d_odata = try cuda.dupe(f32, h_idata[0..num_blocks]);
    defer d_odata.free();

    try reduce(module, size, num_threads, num_blocks, d_idata, d_odata);
    const test_iterations = 100;

    var timer: Timer = .{ .io = init.io };

    const gpu_result = try benchmarkReduce(
        module,
        size,
        num_threads,
        num_blocks,
        settings,
        test_iterations,
        &timer,
        h_odata,
        d_idata,
        d_odata,
    );

    const reduce_time = timer.averageMilliseconds();
    std.log.info("Average time: {d:.6} ms", .{reduce_time});
    std.log.info("Bandwidth:    {d:.6} GB/s", .{
        @as(f64, @floatFromInt(bytes)) / (reduce_time * 1.0e6),
    });

    const cpu_result = reduceCpu(h_idata);

    std.log.info("GPU result = {d:.12}", .{gpu_result});
    std.log.info("CPU result = {d:.12}", .{cpu_result});

    const threshold = 1e-8 * @as(f64, @floatFromInt(size));
    const diff = @abs(@as(f64, gpu_result) - @as(f64, cpu_result));
    return diff < threshold;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{sdk_sample});

    const device = try cuda.init(0);
    defer device.deinit();

    const capability = try device.computeCapability();
    std.log.info("GPU Device supports SM {d}.{d} compute capability", .{
        capability.major,
        capability.minor,
    });

    const module = try cuda.Module.load(@embedFile("threadFenceReduction_kernel.ptx"));
    defer module.unload();

    const test_result = try runTest(init, args, module);
    if (!test_result) std.process.exit(1);
}
