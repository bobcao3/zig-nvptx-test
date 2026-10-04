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

//! This sample calculates scalar products of a given set of input vector pairs.

const std = @import("std");
const cuda = @import("cuda");
const scalarProdCpu = @import("scalarProd_cpu.zig").scalarProdCpu;

/// Returns a uniformly distributed random float in the range [`low`, `high`].
fn randFloat(random: std.Random, low: f32, high: f32) f32 {
    const t = random.float(f32);
    return (1.0 - t) * low + t * high;
}

/// The total number of input vector pairs.
const vector_n = 256;
/// The number of elements per vector. A multiple of the warp size meets the memory coalescing
/// constraints.
const element_n = 4096;
const data_n = vector_n * element_n;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("scalarProd_kernel.ptx"));
    defer module.unload();
    const scalar_prod_gpu = try module.function("scalarProdGpu");

    std.log.info("Initializing data...", .{});
    std.log.info("...allocating CPU memory.", .{});
    const h_a = try init.gpa.alloc(f32, data_n);
    defer init.gpa.free(h_a);
    const h_b = try init.gpa.alloc(f32, data_n);
    defer init.gpa.free(h_b);
    const h_c_cpu = try init.gpa.alloc(f32, vector_n);
    defer init.gpa.free(h_c_cpu);
    const h_c_gpu = try init.gpa.alloc(f32, vector_n);
    defer init.gpa.free(h_c_gpu);

    std.log.info("...allocating GPU memory.", .{});
    const d_a = try cuda.alloc(f32, data_n);
    defer d_a.free();
    const d_b = try cuda.alloc(f32, data_n);
    defer d_b.free();
    const d_c = try cuda.alloc(f32, vector_n);
    defer d_c.free();

    std.log.info("...generating input data in CPU mem.", .{});
    var prng: std.Random.DefaultPrng = .init(123);
    const random = prng.random();
    for (h_a, h_b) |*a, *b| {
        a.* = randFloat(random, 0.0, 1.0);
        b.* = randFloat(random, 0.0, 1.0);
    }

    std.log.info("...copying input data to GPU mem.", .{});
    try d_a.copyFrom(h_a);
    try d_b.copyFrom(h_b);
    std.log.info("Data init done.", .{});

    std.log.info("Executing GPU kernel...", .{});
    try cuda.synchronize();
    const start = std.Io.Clock.awake.now(init.io);
    try scalar_prod_gpu.launch(.{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, .{
        d_c,
        d_a,
        d_b,
        @as(i32, vector_n),
        @as(i32, element_n),
    });
    try cuda.synchronize();
    const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io));
    std.log.info("GPU time: {d:.6} msecs.", .{@as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1.0e6});

    std.log.info("Reading back GPU result...", .{});
    try d_c.copyTo(h_c_gpu);

    std.log.info("Checking GPU results...", .{});
    std.log.info("..running CPU scalar product calculation", .{});
    scalarProdCpu(h_c_cpu, h_a, h_b, vector_n, element_n);

    std.log.info("...comparing the results", .{});
    var sum_delta: f64 = 0;
    var sum_ref: f64 = 0;
    for (h_c_gpu, h_c_cpu) |gpu, cpu| {
        const delta = @abs(gpu - cpu);
        const ref: f64 = cpu;
        sum_delta += delta;
        sum_ref += ref;
    }
    const l1_norm = sum_delta / sum_ref;

    std.log.info("Shutting down...", .{});

    std.log.info("L1 error: {E}", .{l1_norm});
    if (l1_norm < 1e-6) {
        std.log.info("Test passed", .{});
    } else {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }
}
