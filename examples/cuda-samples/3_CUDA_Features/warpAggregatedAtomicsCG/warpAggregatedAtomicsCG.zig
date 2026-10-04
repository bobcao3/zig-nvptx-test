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

const num_elems = 10000000;
const num_threads_per_block = 512;

const Kernels = struct {
    filter_arr: cuda.Function,
    map_to_buckets: cuda.Function,
    calculate_max_in_each_buckets: cuda.Function,
};

fn mapIndicesToBuckets(
    gpa: std.mem.Allocator,
    kernels: Kernels,
    h_src_arr: []const i32,
    d_src_arr: cuda.Buffer(i32),
    num_of_buckets: i32,
) !bool {
    const buckets: usize = @intCast(num_of_buckets);
    const cpu_bucket_counters = try gpa.alloc(i32, buckets);
    defer gpa.free(cpu_bucket_counters);
    const h_bucket_counters = try gpa.alloc(i32, buckets);
    defer gpa.free(h_bucket_counters);

    @memset(cpu_bucket_counters, 0);
    for (h_bucket_counters, 0..) |*counter, i| counter.* = @intCast(i * num_elems);

    const d_indices_buckets = try cuda.alloc(i32, num_elems * buckets);
    defer d_indices_buckets.free();
    const d_bucket_counters = try cuda.alloc(i32, buckets);
    defer d_bucket_counters.free();

    try d_bucket_counters.copyFrom(h_bucket_counters);

    try kernels.map_to_buckets.launch(.{
        .grid = .{ .x = num_elems / num_threads_per_block },
        .block = .{ .x = num_threads_per_block },
    }, .{ d_src_arr, d_indices_buckets, d_bucket_counters, @as(i32, num_elems), num_of_buckets });

    try d_bucket_counters.copyTo(h_bucket_counters);

    for (h_src_arr) |bucket| cpu_bucket_counters[@intCast(bucket)] += 1;

    var all_match = true;
    var final_elems: i32 = 0;
    for (h_bucket_counters, cpu_bucket_counters, 0..) |gpu_counter, cpu_counter, i| {
        const count = gpu_counter - @as(i32, @intCast(i * num_elems));
        final_elems += count;
        if (cpu_counter != count) {
            all_match = false;
            break;
        }
    }

    return all_match or final_elems == num_elems;
}

fn calculateMaxInBuckets(
    gpa: std.mem.Allocator,
    random: std.Random,
    kernels: Kernels,
    h_src_arr: []const i32,
    d_src_arr: cuda.Buffer(i32),
    num_of_buckets: i32,
) !bool {
    const buckets: usize = @intCast(num_of_buckets);
    const h_value_in_buckets = try gpa.alloc(i32, num_elems);
    defer gpa.free(h_value_in_buckets);
    const cpu_buckets_max = try gpa.alloc(i32, buckets);
    defer gpa.free(cpu_buckets_max);
    const h_buckets_max = try gpa.alloc(i32, buckets);
    defer gpa.free(h_buckets_max);

    @memset(cpu_buckets_max, 0);

    for (h_value_in_buckets) |*value| value.* = random.int(u31);

    const d_value_in_buckets = try cuda.alloc(i32, num_elems);
    defer d_value_in_buckets.free();
    const d_buckets_max = try cuda.alloc(i32, buckets);
    defer d_buckets_max.free();

    try d_buckets_max.memset(0);
    try d_value_in_buckets.copyFrom(h_value_in_buckets);

    try kernels.calculate_max_in_each_buckets.launch(.{
        .grid = .{ .x = num_elems / num_threads_per_block },
        .block = .{ .x = num_threads_per_block },
    }, .{ d_src_arr, d_value_in_buckets, d_buckets_max, @as(i32, num_elems), num_of_buckets });

    try d_buckets_max.copyTo(h_buckets_max);

    for (h_src_arr, h_value_in_buckets) |bucket, value| {
        const cpu_max = &cpu_buckets_max[@intCast(bucket)];
        if (cpu_max.* < value) cpu_max.* = value;
    }

    var all_match = true;
    for (cpu_buckets_max, h_buckets_max, 0..) |cpu_max, gpu_max, i| {
        if (cpu_max != gpu_max) {
            all_match = false;
            std.log.err("CPU i={d}  max = {d} mismatches GPU max = {d}", .{ i, cpu_max, gpu_max });
            break;
        }
    }
    if (all_match) {
        std.log.info("CPU max matches GPU max", .{});
    }

    return all_match;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const num_of_buckets = 5;

    const data_to_filter = try gpa.alloc(i32, num_elems);
    defer gpa.free(data_to_filter);

    var prng: std.Random.DefaultPrng = .init(1);
    const random = prng.random();

    for (data_to_filter) |*element| element.* = random.uintLessThan(u31, num_of_buckets);

    const device = try cuda.init(0);
    defer device.deinit();

    const module = try cuda.Module.load(@embedFile("warpAggregatedAtomicsCG_kernel.ptx"));
    defer module.unload();
    const kernels: Kernels = .{
        .filter_arr = try module.function("filterArr"),
        .map_to_buckets = try module.function("mapToBuckets"),
        .calculate_max_in_each_buckets = try module.function("calculateMaxInEachBuckets"),
    };

    const d_data_to_filter = try cuda.dupe(i32, data_to_filter);
    defer d_data_to_filter.free();
    const d_filtered_data = try cuda.alloc(i32, num_elems);
    defer d_filtered_data.free();
    const d_nres = try cuda.alloc(i32, 1);
    defer d_nres.free();

    try d_nres.memset(0);

    try kernels.filter_arr.launch(.{
        .grid = .{ .x = num_elems / num_threads_per_block + 1 },
        .block = .{ .x = num_threads_per_block },
    }, .{ d_filtered_data, d_nres, d_data_to_filter, @as(i32, num_elems) });

    var nres: i32 = 0;
    try d_nres.copyTo((&nres)[0..1]);

    const filtered_data = try gpa.alloc(i32, @intCast(nres));
    defer gpa.free(filtered_data);

    try d_filtered_data.copyTo(filtered_data);

    const host_filtered_data = try gpa.alloc(i32, num_elems);
    defer gpa.free(host_filtered_data);

    var host_flt_count: i32 = 0;
    for (data_to_filter) |element| {
        if (element > 0) {
            host_filtered_data[@intCast(host_flt_count)] = element;
            host_flt_count += 1;
        }
    }

    const major = (try device.computeCapability()).major;

    var map_indices_to_buckets_ok = true;
    var calculate_max_in_buckets_ok = true;
    if (major >= 7) {
        map_indices_to_buckets_ok = try mapIndicesToBuckets(
            gpa,
            kernels,
            data_to_filter,
            d_data_to_filter,
            num_of_buckets,
        );
        calculate_max_in_buckets_ok = try calculateMaxInBuckets(
            gpa,
            random,
            kernels,
            data_to_filter,
            d_data_to_filter,
            num_of_buckets,
        );
    }

    const passed = host_flt_count == nres and map_indices_to_buckets_ok and
        calculate_max_in_buckets_ok;
    std.log.info("Warp Aggregated Atomics {s} ", .{if (passed) "PASSED" else "FAILED"});
    if (!passed) std.process.exit(1);
}
