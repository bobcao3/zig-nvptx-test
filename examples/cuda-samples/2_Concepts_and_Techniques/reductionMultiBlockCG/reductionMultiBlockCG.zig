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

const std = @import("std");
const cuda = @import("cuda");

const sdk_sample = "reductionMultiBlockCG";

fn callReduceSinglePassMultiBlockCg(
    kernel: cuda.Function,
    size: u32,
    threads: u32,
    num_blocks: u32,
    d_idata: cuda.Buffer(f32),
    d_odata: cuda.Buffer(f32),
) !void {
    const smem_size = threads * @sizeOf(f64);

    try kernel.launchCooperative(.{
        .grid = .{ .x = num_blocks },
        .block = .{ .x = threads },
        .shared_memory_bytes = smem_size,
    }, .{ d_idata, d_odata, size });
}

/// Returns the sum of `data` computed with Kahan summation, which keeps the sum of large arrays
/// accurate.
fn reduceCpu(comptime T: type, data: []const T) T {
    var sum = data[0];
    var c: T = 0.0;

    for (data[1..]) |element| {
        const y = element - c;
        const t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

const Blocks = struct { blocks: u32, threads: u32 };

fn getNumBlocksAndThreads(kernel: cuda.Function, n: u32, max_blocks: u32) !Blocks {
    var blocks: u32 = undefined;
    var threads: u32 = undefined;
    if (n == 1) {
        threads = 1;
        blocks = 1;
    } else {
        const sizes = try kernel.maxPotentialBlockSize(0, 0);
        blocks = sizes.min_grid_size;
        threads = sizes.block_size;
    }

    blocks = @min(max_blocks, blocks);
    return .{ .blocks = blocks, .threads = threads };
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

    fn averageMilliseconds(timer: Timer) f64 {
        return 1000.0 * timer.total_seconds / @as(f64, @floatFromInt(timer.laps));
    }
};

/// Performs the reduction of the input data multiple times and measures the average reduction
/// time.
fn benchmarkReduce(
    kernel: cuda.Function,
    n: u32,
    num_threads: u32,
    num_blocks: u32,
    test_iterations: u32,
    timer: *Timer,
    d_idata: cuda.Buffer(f32),
    d_odata: cuda.Buffer(f32),
) !f32 {
    var gpu_result: f32 = 0;

    std.log.info("Launching {s} kernel", .{"SinglePass Multi Block Cooperative Groups"});
    for (0..test_iterations) |_| {
        gpu_result = 0;
        timer.start();
        try callReduceSinglePassMultiBlockCg(kernel, n, num_threads, num_blocks, d_idata, d_odata);
        try cuda.synchronize();
        timer.stop();
    }

    try d_odata.copyTo((&gpu_result)[0..1]);

    return gpu_result;
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

/// Runs the reduction test and returns whether the GPU result matches the CPU result.
fn runTest(init: std.process.Init, args: []const [:0]const u8, device: cuda.Device) !bool {
    const size: u32 = try flagValue(args, "n") orelse 1 << 25;

    std.log.info("{d} elements", .{size});

    const module = try cuda.Module.load(@embedFile("reductionMultiBlockCG_kernel.ptx"));
    defer module.unload();
    const kernel = try module.function("reduceSinglePassMultiBlockCg");

    const h_idata = try init.gpa.alloc(f32, size);
    defer init.gpa.free(h_idata);

    var prng: std.Random.DefaultPrng = .init(1);
    for (h_idata) |*element| {
        const rand_max: f32 = @floatFromInt(@as(i32, std.math.maxInt(i32)));
        element.* = @as(f32, @floatFromInt(prng.random().int(u8))) / rand_max;
    }

    const max_threads_per_block: u32 = @intCast(try device.attribute(.max_threads_per_block));
    const multiprocessor_count: u32 = @intCast(try device.attribute(.multiprocessor_count));
    const max_threads_per_multiprocessor: u32 =
        @intCast(try device.attribute(.max_threads_per_multiprocessor));

    const max_blocks = try flagValue(args, "maxblocks") orelse
        multiprocessor_count * (max_threads_per_multiprocessor / max_threads_per_block);

    var blocks_and_threads = try getNumBlocksAndThreads(kernel, size, max_blocks);
    const num_threads = blocks_and_threads.threads;

    const num_blocks_per_sm = try kernel.maxActiveBlocksPerMultiprocessor(
        num_threads,
        num_threads * @sizeOf(f64),
    );

    const num_sms = multiprocessor_count;
    if (blocks_and_threads.blocks > num_blocks_per_sm * num_sms) {
        blocks_and_threads.blocks = num_blocks_per_sm * num_sms;
    }
    const num_blocks = blocks_and_threads.blocks;
    std.log.info("numThreads: {d}", .{num_threads});
    std.log.info("numBlocks: {d}", .{num_blocks});

    const d_idata = try cuda.dupe(f32, h_idata);
    defer d_idata.free();
    const d_odata = try cuda.dupe(f32, h_idata[0..num_blocks]);
    defer d_odata.free();

    const test_iterations = 100;

    var timer: Timer = .{ .io = init.io };

    const gpu_result = try benchmarkReduce(
        kernel,
        size,
        num_threads,
        num_blocks,
        test_iterations,
        &timer,
        d_idata,
        d_odata,
    );

    const reduce_time = timer.averageMilliseconds();
    std.log.info("Average time: {d:.6} ms", .{reduce_time});
    std.log.info("Bandwidth:    {d:.6} GB/s", .{
        @as(f64, @floatFromInt(size * @sizeOf(i32))) / (reduce_time * 1.0e6),
    });

    const cpu_result = reduceCpu(f32, h_idata);
    std.log.info("GPU result = {d:.12}", .{gpu_result});
    std.log.info("CPU result = {d:.12}", .{cpu_result});

    const threshold = 1e-8 * @as(f64, @floatFromInt(size));
    const diff = @abs(@as(f64, gpu_result) - @as(f64, cpu_result));
    return diff < threshold;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{sdk_sample});

    const device_ordinal = 0;
    const device = try cuda.init(device_ordinal);
    defer device.deinit();

    if (try device.attribute(.cooperative_launch) == 0) {
        std.log.warn("Selected GPU ({d}) does not support Cooperative Kernel Launch, " ++
            "Waiving the run", .{device_ordinal});
        std.process.exit(2);
    }

    if (!try runTest(init, args, device)) std.process.exit(1);
}
