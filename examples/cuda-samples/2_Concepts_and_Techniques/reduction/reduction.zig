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
//! single value.
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
//! - `--n=<N>`: Specify the number of elements to reduce (default 16777216).
//! - `--threads=<N>`: Specify the number of threads per block (default 256).
//! - `--kernel=<N>`: Specify which kernel to run (0-9, default 7).
//! - `--maxblocks=<N>`: Specify the maximum number of thread blocks to launch (kernel 6 and
//!   above, default 64).
//! - `--cpufinal`: Read back the per-block results and do the final sum of block sums on the CPU
//!   (default false).
//! - `--cputhresh=<N>`: The threshold of the number of block sums below which to perform a CPU
//!   final reduction (default 1).
//! - `-type=<T>`: The datatype for the reduction, where T is "int", "float", or "double" (default
//!   int).

const std = @import("std");
const cuda = @import("cuda");
const common = @import("reduction_common.zig");

const ReduceType = enum {
    int,
    float,
    double,

    fn Element(comptime reduce_type: ReduceType) type {
        return switch (reduce_type) {
            .int => i32,
            .float => f32,
            .double => f64,
        };
    }
};

const max_block_dim_size = 65535;

const Limits = struct {
    max_grid_size: u32,
    max_threads_per_block: u32,
};

const Blocks = struct { blocks: u32, threads: u32 };

fn isPow2(x: u32) bool {
    return x & (x -% 1) == 0;
}

/// Integer arithmetic wraps around, as it does on the GPU.
fn plus(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (@typeInfo(@TypeOf(a)) == .int) a +% b else a + b;
}

fn minus(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (@typeInfo(@TypeOf(a)) == .int) a -% b else a - b;
}

/// Returns the sum of `data` computed with Kahan summation, which keeps the sum of large arrays
/// accurate.
fn reduceCpu(comptime T: type, data: []const T) T {
    var sum = data[0];
    var c: T = 0;

    for (data[1..]) |element| {
        const y = minus(element, c);
        const t = plus(sum, y);
        c = minus(minus(t, sum), y);
        sum = t;
    }

    return sum;
}

/// For the kernels 3 and above, the number of work items per work group is the minimum of
/// `max_threads` and n/2. For the kernels below 3, it is the minimum of `max_threads` and n. The
/// kernels 6 and above observe `max_blocks`, because each work item can process a variable number
/// of elements.
fn getNumBlocksAndThreads(
    limits: Limits,
    which_kernel: u32,
    n: u32,
    max_blocks: u32,
    max_threads: u32,
) Blocks {
    var threads: u32 = undefined;
    var blocks: u32 = undefined;
    if (which_kernel < 3) {
        threads = if (n < max_threads) std.math.ceilPowerOfTwoAssert(u32, n) else max_threads;
        blocks = (n + threads - 1) / threads;
    } else {
        threads = if (n < max_threads * 2)
            std.math.ceilPowerOfTwoAssert(u32, (n + 1) / 2)
        else
            max_threads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    const threads_times_blocks = @as(f32, @floatFromInt(threads)) * @as(f32, @floatFromInt(blocks));
    const grid_capacity = @as(f32, @floatFromInt(limits.max_grid_size)) *
        @as(f32, @floatFromInt(limits.max_threads_per_block));
    if (threads_times_blocks > grid_capacity) {
        std.log.err("n is too large, please choose a smaller number!", .{});
    }

    if (blocks > limits.max_grid_size) {
        std.log.warn("Grid size <{d}> exceeds the device capability <{d}>, set block size as " ++
            "{d} (original {d})", .{ blocks, limits.max_grid_size, threads * 2, threads });

        blocks /= 2;
        threads *= 2;
    }

    if (which_kernel >= 6) blocks = @min(max_blocks, blocks);

    return .{ .blocks = blocks, .threads = threads };
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

/// Launches the reduction kernel `which_kernel`.
fn reduce(
    comptime T: type,
    module: cuda.Module,
    size: u32,
    threads: u32,
    blocks: u32,
    which_kernel_in: u32,
    d_idata: cuda.Buffer(T),
    d_odata: cuda.Buffer(T),
) !void {
    var config: cuda.LaunchConfig = .{
        .grid = .{ .x = blocks },
        .block = .{ .x = threads },
        // With only one warp per block, two warps' worth of shared memory keep the shared memory
        // accesses in bounds.
        .shared_memory_bytes = (if (threads <= 32) 2 * threads else threads) * @sizeOf(T),
    };
    const args = .{ d_idata, d_odata, size };

    // Kernel 9 cannot work for more than 64 threads, so kernel 7 replaces it.
    const which_kernel = if (threads < 64 and which_kernel_in == 9) 7 else which_kernel_in;

    switch (which_kernel) {
        inline 0...3 => |kernel| try launchKernel(module, common.kernelName(kernel, T, null, null), config, args),
        inline 4, 5 => |kernel| inline for (common.block_sizes) |block_size| {
            if (threads == block_size) {
                try launchKernel(module, common.kernelName(kernel, T, block_size, null), config, args);
            }
        },
        6 => switch (isPow2(size)) {
            inline else => |n_is_pow2| inline for (common.block_sizes) |block_size| {
                if (threads == block_size) {
                    try launchKernel(module, common.kernelName(6, T, block_size, n_is_pow2), config, args);
                }
            },
        },
        7 => {
            // Kernel 7 needs only one element of shared memory per warp.
            config.shared_memory_bytes = ((threads / 32) + 1) * @sizeOf(T);
            switch (isPow2(size)) {
                inline else => |n_is_pow2| inline for (common.block_sizes_7) |block_size| {
                    if (threads == block_size) {
                        try launchKernel(module, common.kernelName(7, T, block_size, n_is_pow2), config, args);
                    }
                },
            }
        },
        8 => try launchKernel(module, common.kernelName(8, T, null, null), config, args),
        9 => {
            config.shared_memory_bytes = common.multi_warp_groups * @sizeOf(T);
            inline for (common.multi_warp_block_sizes) |block_size| {
                if (threads == block_size) {
                    try launchKernel(module, common.kernelName(9, T, block_size, null), config, args);
                    return;
                }
            }
            std.log.err("thread block size of < 64 is not supported for this kernel", .{});
        },
        else => {},
    }
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
    which_kernel: u32,
    cpu_final_reduction: bool,
    cpu_final_threshold: u32,
};

/// Performs the reduction of the input data `test_iterations` times and measures the average
/// reduction time.
fn benchmarkReduce(
    comptime T: type,
    module: cuda.Module,
    limits: Limits,
    n: u32,
    blocks_and_threads: Blocks,
    settings: Settings,
    test_iterations: u32,
    timer: *Timer,
    h_odata: []T,
    d_idata: cuda.Buffer(T),
    d_odata: cuda.Buffer(T),
) !T {
    const num_blocks = blocks_and_threads.blocks;
    var gpu_result: T = 0;
    var need_read_back = true;

    const d_intermediate_sums = try cuda.alloc(T, num_blocks);
    defer d_intermediate_sums.free();

    for (0..test_iterations) |_| {
        gpu_result = 0;

        try cuda.synchronize();
        timer.start();

        try reduce(T, module, n, blocks_and_threads.threads, num_blocks, settings.which_kernel, d_idata, d_odata);

        if (settings.cpu_final_reduction) {
            try d_odata.copyTo(h_odata[0..num_blocks]);

            for (h_odata[0..num_blocks]) |partial_sum| gpu_result = plus(gpu_result, partial_sum);

            need_read_back = false;
        } else {
            var s = num_blocks;
            const kernel = settings.which_kernel;

            while (s > settings.cpu_final_threshold) {
                const next = getNumBlocksAndThreads(limits, kernel, s, settings.max_blocks, settings.max_threads);
                var partial_sums = d_odata;
                partial_sums.len = s;
                try d_intermediate_sums.copyFromBuffer(partial_sums);
                try reduce(T, module, s, next.threads, next.blocks, kernel, d_intermediate_sums, d_odata);

                if (kernel < 3) {
                    s = (s + next.threads - 1) / next.threads;
                } else {
                    s = (s + (next.threads * 2 - 1)) / (next.threads * 2);
                }
            }

            if (s > 1) {
                try d_odata.copyTo(h_odata[0..s]);

                for (h_odata[0..s]) |partial_sum| gpu_result = plus(gpu_result, partial_sum);

                need_read_back = false;
            }
        }

        try cuda.synchronize();
        timer.stop();
    }

    if (need_read_back) try d_odata.copyTo((&gpu_result)[0..1]);
    return gpu_result;
}

fn randomElement(comptime T: type, random: std.Random) T {
    const byte = random.int(u8);
    return switch (@typeInfo(T)) {
        .int => byte,
        else => @as(T, @floatFromInt(byte)) / @as(T, @floatFromInt(@as(i32, std.math.maxInt(i32)))),
    };
}

/// Calls `benchmarkReduce` for a range of array sizes and prints a report in comma-separated
/// value format that can be used for generating a "shmoo" plot showing the performance of each
/// kernel for each input size.
fn shmoo(
    comptime T: type,
    gpa: std.mem.Allocator,
    io: std.Io,
    module: cuda.Module,
    limits: Limits,
    min_n: u32,
    max_n: u32,
    max_threads: u32,
    max_blocks: u32,
) !void {
    const h_idata = try gpa.alloc(T, max_n);
    defer gpa.free(h_idata);

    var prng: std.Random.DefaultPrng = .init(2009);
    // Keep the numbers small so we don't get truncation error in the sum
    for (h_idata) |*element| element.* = randomElement(T, prng.random());

    const max_num_blocks = @min(max_n / max_threads, max_block_dim_size);

    const h_odata = try gpa.alloc(T, max_num_blocks);
    defer gpa.free(h_odata);

    const d_idata = try cuda.dupe(T, h_idata);
    defer d_idata.free();
    const d_odata = try cuda.dupe(T, h_idata[0..max_num_blocks]);
    defer d_odata.free();

    for (0..8) |kernel| {
        try reduce(T, module, max_n, max_threads, max_num_blocks, @intCast(kernel), d_idata, d_odata);
    }

    const test_iterations = 100;

    var timer: Timer = .{ .io = io };

    std.log.info("Time in milliseconds for various numbers of elements for each kernel", .{});
    var header: std.Io.Writer.Allocating = .init(gpa);
    defer header.deinit();
    try header.writer.writeAll("Kernel");

    var i = min_n;
    while (i <= max_n) : (i *= 2) try header.writer.print(", {d}", .{i});
    std.log.info("{s}", .{header.written()});

    for (0..8) |kernel| {
        var row: std.Io.Writer.Allocating = .init(gpa);
        defer row.deinit();
        try row.writer.print("{d}", .{kernel});

        i = min_n;
        while (i <= max_n) : (i *= 2) {
            timer.reset();
            const blocks_and_threads = getNumBlocksAndThreads(limits, @intCast(kernel), i, max_blocks, max_threads);

            var reduce_time: f64 = undefined;

            if (blocks_and_threads.blocks <= max_block_dim_size) {
                const settings: Settings = .{
                    .max_threads = max_threads,
                    .max_blocks = max_blocks,
                    .which_kernel = @intCast(kernel),
                    .cpu_final_reduction = false,
                    .cpu_final_threshold = 1,
                };
                _ = try benchmarkReduce(T, module, limits, i, blocks_and_threads, settings, test_iterations, &timer, h_odata, d_idata, d_odata);
                reduce_time = timer.averageMilliseconds();
            } else {
                reduce_time = -1.0;
            }

            try row.writer.print(", {d:.5}", .{reduce_time});
        }
        std.log.info("{s}", .{row.written()});
    }
}

/// Runs the reduction test and returns whether the GPU result matches the CPU result.
fn runTest(
    comptime T: type,
    init: std.process.Init,
    args: []const [:0]const u8,
    module: cuda.Module,
    limits: Limits,
    datatype: ReduceType,
) !bool {
    var size: u32 = 1 << 24;
    var max_threads: u32 = 256;
    var settings: Settings = .{
        .max_threads = 256,
        .max_blocks = 64,
        .which_kernel = 7,
        .cpu_final_reduction = false,
        .cpu_final_threshold = 1,
    };

    if (try flagValue(args, "n")) |value| size = value;
    if (try flagValue(args, "threads")) |value| max_threads = value;
    if (try flagValue(args, "kernel")) |value| settings.which_kernel = value;
    if (try flagValue(args, "maxblocks")) |value| settings.max_blocks = value;
    settings.max_threads = max_threads;

    std.log.info("{d} elements", .{size});
    std.log.info("{d} threads (max)", .{max_threads});

    settings.cpu_final_reduction = hasFlag(args, "cpufinal");

    if (try flagValue(args, "cputhresh")) |value| settings.cpu_final_threshold = value;

    if (hasFlag(args, "shmoo")) {
        try shmoo(T, init.gpa, init.io, module, limits, 1, 33554432, max_threads, settings.max_blocks);
        return true;
    }

    const bytes = size * @sizeOf(T);

    const h_idata = try init.gpa.alloc(T, size);
    defer init.gpa.free(h_idata);

    var prng: std.Random.DefaultPrng = .init(2009);
    // Keep the numbers small so we don't get truncation error in the sum
    for (h_idata) |*element| element.* = randomElement(T, prng.random());

    const blocks_and_threads = getNumBlocksAndThreads(limits, settings.which_kernel, size, settings.max_blocks, max_threads);
    const num_blocks = blocks_and_threads.blocks;
    const num_threads = blocks_and_threads.threads;

    if (num_blocks == 1) settings.cpu_final_threshold = 1;

    const h_odata = try init.gpa.alloc(T, num_blocks);
    defer init.gpa.free(h_odata);

    std.log.info("{d} blocks", .{num_blocks});

    const d_idata = try cuda.dupe(T, h_idata);
    defer d_idata.free();
    const d_odata = try cuda.dupe(T, h_idata[0..num_blocks]);
    defer d_odata.free();

    try reduce(T, module, size, num_threads, num_blocks, settings.which_kernel, d_idata, d_odata);

    const test_iterations = 100;

    var timer: Timer = .{ .io = init.io };

    const gpu_result = try benchmarkReduce(T, module, limits, size, blocks_and_threads, settings, test_iterations, &timer, h_odata, d_idata, d_odata);

    const reduce_time = timer.averageMilliseconds() * 1e-3;
    std.log.info("Reduction, Throughput = {d:.4} GB/s, Time = {d:.5} s, Size = {d} Elements, " ++
        "NumDevsUsed = {d}, Workgroup = {d}", .{
        1.0e-9 * @as(f64, @floatFromInt(bytes)) / reduce_time,
        reduce_time,
        size,
        1,
        num_threads,
    });

    const cpu_result = reduceCpu(T, h_idata);

    switch (datatype) {
        .int => {
            std.log.info("GPU result = {d}", .{gpu_result});
            std.log.info("CPU result = {d}", .{cpu_result});
            return gpu_result == cpu_result;
        },
        .float, .double => {
            const precision: usize, const threshold: f64 = switch (datatype) {
                .float => .{ 8, 1e-8 * @as(f64, @floatFromInt(size)) },
                else => .{ 12, 1e-12 * @as(f64, @floatFromInt(size)) },
            };

            std.log.info("GPU result = {[value]d:.[precision]}", .{ .value = @as(f64, gpu_result), .precision = precision });
            std.log.info("CPU result = {[value]d:.[precision]}", .{ .value = @as(f64, cpu_result), .precision = precision });

            const diff = @abs(@as(f64, gpu_result) - @as(f64, cpu_result));
            return diff < threshold;
        },
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

fn flagString(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') return flag[name.len + 1 ..];
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

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    var datatype: ReduceType = .int;
    if (flagString(args, "type")) |type_input| {
        if (std.ascii.eqlIgnoreCase(type_input, "float")) {
            datatype = .float;
        } else if (std.ascii.eqlIgnoreCase(type_input, "double")) {
            datatype = .double;
        } else if (!std.ascii.eqlIgnoreCase(type_input, "int")) {
            std.log.warn("Type {s} is not recognized. Using default type int.", .{type_input});
        }
    }

    const ordinal: c_int = @intCast(try flagValue(args, "device") orelse 0);
    const device = try cuda.init(ordinal);
    defer device.deinit();

    var name_buffer: [256]u8 = undefined;
    std.log.info("Using Device {d}: {s}", .{ ordinal, try device.name(&name_buffer) });

    std.log.info("Reducing array of type {t}", .{datatype});

    const limits: Limits = .{
        .max_grid_size = @intCast(try device.attribute(.max_grid_dim_x)),
        .max_threads_per_block = @intCast(try device.attribute(.max_threads_per_block)),
    };

    const module = try cuda.Module.load(@embedFile("reduction_kernel.ptx"));
    defer module.unload();

    const passed = switch (datatype) {
        inline else => |reduce_type| try runTest(reduce_type.Element(), init, args, module, limits, reduce_type),
    };

    if (passed) std.log.info("Test passed", .{}) else std.log.err("Test failed!", .{});
    if (!passed) std.process.exit(1);
}
