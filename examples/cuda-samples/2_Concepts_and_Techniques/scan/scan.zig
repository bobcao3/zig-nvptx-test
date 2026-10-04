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
const common = @import("scan_common.zig");
const scanExclusiveHost = @import("scan_gold.zig").scanExclusiveHost;

const threadblock_size = common.threadblock_size;

const Scan = struct {
    module: cuda.Module,
    scan_exclusive_shared: cuda.Function,
    scan_exclusive_shared2: cuda.Function,
    uniform_update: cuda.Function,
    d_buf: cuda.Buffer(u32),
};

fn initScan() !Scan {
    const module = try cuda.Module.load(@embedFile("scan_kernel.ptx"));
    errdefer module.unload();
    return .{
        .module = module,
        .scan_exclusive_shared = try module.function("scanExclusiveShared"),
        .scan_exclusive_shared2 = try module.function("scanExclusiveShared2"),
        .uniform_update = try module.function("uniformUpdate"),
        .d_buf = try cuda.alloc(u32, common.max_batch_elements / (4 * threadblock_size)),
    };
}

fn closeScan(scan: Scan) void {
    scan.d_buf.free();
    scan.module.unload();
}

fn factorRadix2(log2_l: *u32, l_in: u32) u32 {
    var l = l_in;
    if (l == 0) {
        log2_l.* = 0;
        return 0;
    }
    log2_l.* = 0;
    while (l & 1 == 0) : (l >>= 1) log2_l.* += 1;
    return l;
}

fn scanExclusiveShort(
    scan: Scan,
    d_dst: cuda.Buffer(u32),
    d_src: cuda.Buffer(u32),
    batch_size: u32,
    array_length: u32,
) !usize {
    var log2_l: u32 = undefined;
    const factorization_remainder = factorRadix2(&log2_l, array_length);
    std.debug.assert(factorization_remainder == 1);

    std.debug.assert(array_length >= common.min_short_array_size and
        array_length <= common.max_short_array_size);
    std.debug.assert(batch_size * array_length <= common.max_batch_elements);
    std.debug.assert(batch_size * array_length % (4 * threadblock_size) == 0);

    try scan.scan_exclusive_shared.launch(.{
        .grid = .{ .x = (batch_size * array_length) / (4 * threadblock_size) },
        .block = .{ .x = threadblock_size },
    }, .{ d_dst, d_src, array_length });

    return threadblock_size;
}

fn scanExclusiveLarge(
    scan: Scan,
    d_dst: cuda.Buffer(u32),
    d_src: cuda.Buffer(u32),
    batch_size: u32,
    array_length: u32,
) !usize {
    var log2_l: u32 = undefined;
    const factorization_remainder = factorRadix2(&log2_l, array_length);
    std.debug.assert(factorization_remainder == 1);

    std.debug.assert(array_length >= common.min_large_array_size and
        array_length <= common.max_large_array_size);
    std.debug.assert(batch_size * array_length <= common.max_batch_elements);

    const block_count = (batch_size * array_length) / (4 * threadblock_size);
    try scan.scan_exclusive_shared.launch(.{
        .grid = .{ .x = block_count },
        .block = .{ .x = threadblock_size },
    }, .{ d_dst, d_src, @as(u32, 4 * threadblock_size) });

    // Not all work groups need to be packed with input data. The inactive work items of the
    // highest work group skip the global reads and writes.
    const block_count2 = std.math.divCeil(u32, block_count, threadblock_size) catch unreachable;
    try scan.scan_exclusive_shared2.launch(.{
        .grid = .{ .x = block_count2 },
        .block = .{ .x = threadblock_size },
    }, .{ scan.d_buf, d_dst, d_src, block_count, array_length / (4 * threadblock_size) });

    try scan.uniform_update.launch(.{
        .grid = .{ .x = block_count },
        .block = .{ .x = threadblock_size },
    }, .{ d_dst, scan.d_buf });

    return threadblock_size;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();

    const n: u32 = 13 * 1048576 / 2;

    std.log.info("Allocating and initializing host arrays...", .{});
    const h_input = try init.gpa.alloc(u32, n);
    defer init.gpa.free(h_input);
    const h_output_cpu = try init.gpa.alloc(u32, n);
    defer init.gpa.free(h_output_cpu);
    const h_output_gpu = try init.gpa.alloc(u32, n);
    defer init.gpa.free(h_output_gpu);
    var prng: std.Random.DefaultPrng = .init(2009);
    for (h_input) |*element| element.* = prng.random().int(u31);

    std.log.info("Allocating and initializing CUDA arrays...", .{});
    const d_input = try cuda.dupe(u32, h_input);
    defer d_input.free();
    const d_output = try cuda.alloc(u32, n);
    defer d_output.free();

    std.log.info("Initializing CUDA-C scan...", .{});
    const scan = try initScan();
    defer closeScan(scan);

    var global_flag = true;
    var sz_workgroup: usize = undefined;
    const i_cycles = 100;
    std.log.info("*** Running GPU scan for short arrays ({d} identical iterations)...", .{i_cycles});

    var array_length = common.min_short_array_size;
    while (array_length <= common.max_short_array_size) : (array_length <<= 1) {
        std.log.info("Running scan for {d} elements ({d} arrays)...", .{ array_length, n / array_length });
        try cuda.synchronize();
        const start = std.Io.Clock.awake.now(init.io);

        for (0..i_cycles) |_| {
            sz_workgroup = try scanExclusiveShort(scan, d_output, d_input, n / array_length, array_length);
        }

        try cuda.synchronize();
        const timer_value = elapsedSeconds(init.io, start) / i_cycles;

        std.log.info("Validating the results...", .{});
        std.log.info("...reading back GPU results", .{});
        try d_output.copyTo(h_output_gpu);

        std.log.info(" ...scanExclusiveHost()", .{});
        scanExclusiveHost(h_output_cpu, h_input, n / array_length, array_length);

        std.log.info(" ...comparing the results", .{});
        const local_flag = std.mem.eql(u32, h_output_cpu, h_output_gpu);

        if (local_flag) std.log.info(" ...Results Match", .{}) else std.log.err(" ...Results DON'T Match !!!", .{});
        global_flag = global_flag and local_flag;

        if (array_length == common.max_short_array_size) {
            printThroughput(array_length, timer_value, sz_workgroup);
        }
    }

    std.log.info("***Running GPU scan for large arrays ({d} identical iterations)...", .{i_cycles});

    array_length = common.min_large_array_size;
    while (array_length <= common.max_large_array_size) : (array_length <<= 1) {
        std.log.info("Running scan for {d} elements ({d} arrays)...", .{ array_length, n / array_length });
        try cuda.synchronize();
        const start = std.Io.Clock.awake.now(init.io);

        for (0..i_cycles) |_| {
            sz_workgroup = try scanExclusiveLarge(scan, d_output, d_input, n / array_length, array_length);
        }

        try cuda.synchronize();
        const timer_value = elapsedSeconds(init.io, start) / i_cycles;

        std.log.info("Validating the results...", .{});
        std.log.info("...reading back GPU results", .{});
        try d_output.copyTo(h_output_gpu);

        std.log.info("...scanExclusiveHost()", .{});
        scanExclusiveHost(h_output_cpu, h_input, n / array_length, array_length);

        std.log.info(" ...comparing the results", .{});
        const local_flag = std.mem.eql(u32, h_output_cpu, h_output_gpu);

        if (local_flag) std.log.info(" ...Results Match", .{}) else std.log.err(" ...Results DON'T Match !!!", .{});
        global_flag = global_flag and local_flag;

        if (array_length == common.max_large_array_size) {
            printThroughput(array_length, timer_value, sz_workgroup);
        }
    }

    std.log.info("Shutting down...", .{});

    if (!global_flag) std.process.exit(1);
}

fn elapsedSeconds(io: std.Io, start: std.Io.Timestamp) f64 {
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io));
    return @as(f64, @floatFromInt(elapsed.toNanoseconds())) * 1.0e-9;
}

fn printThroughput(array_length: u32, timer_value: f64, sz_workgroup: usize) void {
    std.log.info("scan, Throughput = {d:.4} MElements/s, Time = {d:.5} s, Size = {d} " ++
        "Elements, NumDevsUsed = {d}, Workgroup = {d}", .{
        1.0e-6 * @as(f64, @floatFromInt(array_length)) / timer_value,
        timer_value,
        array_length,
        1,
        sz_workgroup,
    });
}
