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

//! CUDA Sorting Networks.
//!
//! This sample implements bitonic sort and odd-even merge sort (also known as Batcher's sort),
//! algorithms belonging to the class of sorting networks. While generally subefficient, for large
//! sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort
//! or radix sort), this may be the preferred algorithms of choice for sorting batches of
//! short-sized to mid-sized (key, value) array pairs. Refer to an excellent tutorial by H. W. Lang
//! https://hwlang.de/algorithmen/sortieren/bitonic/bitonicen.htm

const std = @import("std");
const cuda = @import("cuda");
const common = @import("sortingNetworks_common.zig");

const shared_size_limit = common.shared_size_limit;
const Buffer = cuda.Buffer(u32);

/// Returns the odd factor of `l` and writes the number of factors of two to `log2_l`.
fn factorRadix2(log2_l: *u32, l: u32) u32 {
    if (l == 0) {
        log2_l.* = 0;
        return 0;
    }
    var rest = l;
    log2_l.* = 0;
    while (rest & 1 == 0) : (rest >>= 1) log2_l.* += 1;
    return rest;
}

/// Sorts `batch_size` arrays of `array_length` pairs and returns the work group size.
fn bitonicSort(
    module: cuda.Module,
    d_dst_key: Buffer,
    d_dst_val: Buffer,
    d_src_key: Buffer,
    d_src_val: Buffer,
    batch_size: u32,
    array_length: u32,
    direction: u32,
) !u32 {
    if (array_length < 2) return 0;

    var log2_l: u32 = undefined;
    const factorization_remainder = factorRadix2(&log2_l, array_length);
    std.debug.assert(factorization_remainder == 1);

    const dir: u32 = @intFromBool(direction != 0);

    const block_count = batch_size * array_length / shared_size_limit;
    const thread_count = shared_size_limit / 2;
    const shared_config: cuda.LaunchConfig = .{
        .grid = .{ .x = block_count },
        .block = .{ .x = thread_count },
    };

    if (array_length <= shared_size_limit) {
        std.debug.assert((batch_size * array_length) % shared_size_limit == 0);
        const bitonic_sort_shared = try module.function("bitonicSortShared");
        try bitonic_sort_shared.launch(shared_config, .{
            d_dst_key, d_dst_val, d_src_key, d_src_val, array_length, dir,
        });
    } else {
        const bitonic_sort_shared1 = try module.function("bitonicSortShared1");
        const bitonic_merge_global = try module.function("bitonicMergeGlobal");
        const bitonic_merge_shared = try module.function("bitonicMergeShared");
        try bitonic_sort_shared1.launch(shared_config, .{ d_dst_key, d_dst_val, d_src_key, d_src_val });

        var size: u32 = 2 * shared_size_limit;
        while (size <= array_length) : (size <<= 1) {
            var stride = size / 2;
            while (stride > 0) : (stride >>= 1) {
                if (stride >= shared_size_limit) {
                    try bitonic_merge_global.launch(.{
                        .grid = .{ .x = (batch_size * array_length) / 512 },
                        .block = .{ .x = 256 },
                    }, .{ d_dst_key, d_dst_val, d_dst_key, d_dst_val, array_length, size, stride, dir });
                } else {
                    try bitonic_merge_shared.launch(shared_config, .{
                        d_dst_key, d_dst_val, d_dst_key, d_dst_val, array_length, size, dir,
                    });
                    break;
                }
            }
        }
    }

    return thread_count;
}

fn oddEvenMergeSort(
    module: cuda.Module,
    d_dst_key: Buffer,
    d_dst_val: Buffer,
    d_src_key: Buffer,
    d_src_val: Buffer,
    batch_size: u32,
    array_length: u32,
    direction: u32,
) !void {
    if (array_length < 2) return;

    var log2_l: u32 = undefined;
    const factorization_remainder = factorRadix2(&log2_l, array_length);
    std.debug.assert(factorization_remainder == 1);

    const dir: u32 = @intFromBool(direction != 0);

    const block_count = (batch_size * array_length) / shared_size_limit;
    const thread_count = shared_size_limit / 2;
    const shared_config: cuda.LaunchConfig = .{
        .grid = .{ .x = block_count },
        .block = .{ .x = thread_count },
    };
    const odd_even_merge_sort_shared = try module.function("oddEvenMergeSortShared");

    if (array_length <= shared_size_limit) {
        std.debug.assert(shared_size_limit % array_length == 0);
        try odd_even_merge_sort_shared.launch(shared_config, .{
            d_dst_key, d_dst_val, d_src_key, d_src_val, array_length, dir,
        });
    } else {
        try odd_even_merge_sort_shared.launch(shared_config, .{
            d_dst_key, d_dst_val, d_src_key, d_src_val, @as(u32, shared_size_limit), dir,
        });

        const odd_even_merge_global = try module.function("oddEvenMergeGlobal");
        var size: u32 = 2 * shared_size_limit;
        while (size <= array_length) : (size <<= 1) {
            var stride = size / 2;
            while (stride > 0) : (stride >>= 1) {
                try odd_even_merge_global.launch(.{
                    .grid = .{ .x = (batch_size * array_length) / 512 },
                    .block = .{ .x = 256 },
                }, .{ d_dst_key, d_dst_val, d_dst_key, d_dst_val, array_length, size, stride, dir });
            }
        }
    }
}

/// Validates the sorted keys array (checks for integrity and proper order).
fn validateSortedKeys(
    gpa: std.mem.Allocator,
    res_key: []const u32,
    src_key: []const u32,
    batch_size: u32,
    array_length: u32,
    num_values: u32,
    dir: u32,
) !bool {
    if (array_length < 2) {
        std.log.info("validateSortedKeys(): arrayLength too short, exiting...", .{});
        return true;
    }

    std.log.info("...inspecting keys array: ", .{});

    const src_hist = try gpa.alloc(u32, num_values);
    defer gpa.free(src_hist);
    const res_hist = try gpa.alloc(u32, num_values);
    defer gpa.free(res_hist);

    var flag = true;

    check: for (0..batch_size) |j| {
        const src = src_key[j * array_length ..][0..array_length];
        const res = res_key[j * array_length ..][0..array_length];

        @memset(src_hist, 0);
        @memset(res_hist, 0);

        for (src, res) |src_element, res_element| {
            if (src_element < num_values and res_element < num_values) {
                src_hist[src_element] += 1;
                res_hist[res_element] += 1;
            } else {
                flag = false;
                break;
            }
        }

        if (!flag) {
            std.log.err("***Set {d} source/result key arrays are not limited properly***", .{j});
            break :check;
        }

        if (!std.mem.eql(u32, src_hist, res_hist)) {
            flag = false;
            std.log.err("***Set {d} source/result keys histograms do not match***", .{j});
            break :check;
        }

        for (res[0 .. array_length - 1], res[1..]) |current, next| {
            const ordered = if (dir != 0) next >= current else next <= current;
            if (!ordered) {
                flag = false;
                break;
            }
        }

        if (!flag) {
            std.log.err("***Set {d} result key array is not ordered properly***", .{j});
            break :check;
        }
    }

    if (flag) std.log.info("OK", .{});

    return flag;
}

fn validateValues(
    res_key: []const u32,
    res_val: []const u32,
    src_key: []const u32,
    batch_size: u32,
    array_length: u32,
) bool {
    var correct_flag = true;
    var stable_flag = true;

    std.log.info("...inspecting keys and values array: ", .{});

    for (0..batch_size) |i| {
        const keys = res_key[i * array_length ..][0..array_length];
        const vals = res_val[i * array_length ..][0..array_length];
        for (keys, vals, 0..) |key, val, j| {
            if (key != src_key[val]) correct_flag = false;

            if (j < array_length - 1 and key == keys[j + 1] and val > vals[j + 1]) stable_flag = false;
        }
    }

    if (correct_flag) std.log.info("OK", .{}) else std.log.err("***corrupted!!!***", .{});
    if (stable_flag) {
        std.log.info("...stability property: stable!", .{});
    } else {
        std.log.warn("...stability property: NOT stable", .{});
    }

    return correct_flag;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const gpa = init.gpa;
    std.log.info("{s} Starting...", .{args[0]});

    std.log.info("Starting up CUDA context...", .{});
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("bitonicSort_kernel.ptx"));
    defer module.unload();

    const n = 1048576;
    const dir = 0;
    const num_values = 65536;
    const num_iterations = 1;

    std.log.info("Allocating and initializing host arrays...", .{});
    const h_input_key = try gpa.alloc(u32, n);
    defer gpa.free(h_input_key);
    const h_input_val = try gpa.alloc(u32, n);
    defer gpa.free(h_input_val);
    const h_output_key_gpu = try gpa.alloc(u32, n);
    defer gpa.free(h_output_key_gpu);
    const h_output_val_gpu = try gpa.alloc(u32, n);
    defer gpa.free(h_output_val_gpu);
    var prng: std.Random.DefaultPrng = .init(2001);

    for (h_input_key, h_input_val, 0..) |*key, *val, i| {
        key.* = prng.random().uintLessThan(u32, num_values);
        val.* = @intCast(i);
    }

    std.log.info("Allocating and initializing CUDA arrays...", .{});
    const d_input_key = try cuda.dupe(u32, h_input_key);
    defer d_input_key.free();
    const d_input_val = try cuda.dupe(u32, h_input_val);
    defer d_input_val.free();
    const d_output_key = try cuda.alloc(u32, n);
    defer d_output_key.free();
    const d_output_val = try cuda.alloc(u32, n);
    defer d_output_val.free();

    var flag = true;
    std.log.info("Running GPU bitonic sort ({d} identical iterations)...", .{num_iterations});

    var array_length: u32 = 64;
    while (array_length <= n) : (array_length *= 2) {
        std.log.info("Testing array length {d} ({d} arrays per batch)...", .{ array_length, n / array_length });
        try cuda.synchronize();

        const start = std.Io.Clock.awake.now(init.io);
        var thread_count: u32 = 0;

        for (0..num_iterations) |_| {
            thread_count = try bitonicSort(
                module,
                d_output_key,
                d_output_val,
                d_input_key,
                d_input_val,
                n / array_length,
                array_length,
                dir,
            );
        }

        try cuda.synchronize();

        const elapsed_ns: f64 = @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds());
        const timer_ms = elapsed_ns * 1.0e-6;
        std.log.info("Average time: {d:.6} ms", .{timer_ms / num_iterations});

        if (array_length == n) {
            const time_secs = 1.0e-3 * timer_ms / num_iterations;
            std.log.info(
                "sortingNetworks-bitonic, Throughput = {d:.4} MElements/s, Time = {d:.5} s, " ++
                    "Size = {d} elements, NumDevsUsed = {d}, Workgroup = {d}",
                .{ 1.0e-6 * @as(f64, @floatFromInt(array_length)) / time_secs, time_secs, array_length, 1, thread_count },
            );
        }

        std.log.info("Validating the results...", .{});
        std.log.info("...reading back GPU results", .{});
        try d_output_key.copyTo(h_output_key_gpu);
        try d_output_val.copyTo(h_output_val_gpu);

        const keys_flag = try validateSortedKeys(gpa, h_output_key_gpu, h_input_key, n / array_length, array_length, num_values, dir);
        const values_flag = validateValues(h_output_key_gpu, h_output_val_gpu, h_input_key, n / array_length, array_length);
        flag = flag and keys_flag and values_flag;
    }

    std.log.info("Shutting down...", .{});

    if (!flag) std.process.exit(1);
}
