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

//! This sample evaluates the fair call and put prices for a given set of European options by the
//! Black-Scholes formula.

const std = @import("std");
const cuda = @import("cuda");

/// Polynomial approximation of the cumulative normal distribution function.
fn cnd(d: f64) f64 {
    const a1 = 0.31938153;
    const a2 = -0.356563782;
    const a3 = 1.781477937;
    const a4 = -1.821255978;
    const a5 = 1.330274429;
    const rsqrt_2pi = 0.39894228040143267793994605993438;

    const k = 1.0 / (1.0 + 0.2316419 * @abs(d));

    const polynomial = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))));
    var result = rsqrt_2pi * @exp(-0.5 * d * d) * polynomial;

    if (d > 0) result = 1.0 - result;

    return result;
}

const Prices = struct { call: f32, put: f32 };

/// Black-Scholes formula for both call and put. `sf` is the stock price, `xf` the option strike,
/// `tf` the option years, `rf` the riskless rate and `vf` the volatility rate.
fn blackScholesBodyCpu(sf: f32, xf: f32, tf: f32, rf: f32, vf: f32) Prices {
    const s: f64 = sf;
    const x: f64 = xf;
    const t: f64 = tf;
    const r: f64 = rf;
    const v: f64 = vf;

    const sqrt_t = @sqrt(t);
    const d1 = (@log(s / x) + (r + 0.5 * v * v) * t) / (v * sqrt_t);
    const d2 = d1 - v * sqrt_t;
    const cnd_d1 = cnd(d1);
    const cnd_d2 = cnd(d2);

    // Calculates call and put simultaneously.
    const exp_rt = @exp(-r * t);
    return .{
        .call = @floatCast(s * cnd_d1 - x * exp_rt * cnd_d2),
        .put = @floatCast(x * exp_rt * (1.0 - cnd_d2) - s * (1.0 - cnd_d1)),
    };
}

/// Processes an array of options.
fn blackScholesCpu(
    h_call_result: []f32,
    h_put_result: []f32,
    h_stock_price: []const f32,
    h_option_strike: []const f32,
    h_option_years: []const f32,
    riskfree: f32,
    volatility: f32,
) void {
    for (
        h_call_result,
        h_put_result,
        h_stock_price,
        h_option_strike,
        h_option_years,
    ) |*call, *put, s, x, t| {
        const prices = blackScholesBodyCpu(s, x, t, riskfree, volatility);
        call.* = prices.call;
        put.* = prices.put;
    }
}

/// Returns a uniformly distributed random float in the range from `low` to `high`.
fn randFloat(random: std.Random, low: f32, high: f32) f32 {
    const t = random.float(f32);
    return (1.0 - t) * low + t * high;
}

const opt_n = 4000000;
const num_iterations = 512;

const riskfree_rate = 0.02;
const volatility_rate = 0.30;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("[BlackScholes] - Starting...", .{});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("BlackScholes_kernel.ptx"));
    defer module.unload();
    const black_scholes_gpu = try module.function("blackScholesGpu");

    std.log.info("Initializing data...", .{});
    std.log.info("...allocating CPU memory for options.", .{});
    const h_call_result_cpu = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_call_result_cpu);
    const h_put_result_cpu = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_put_result_cpu);
    const h_call_result_gpu = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_call_result_gpu);
    const h_put_result_gpu = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_put_result_gpu);
    const h_stock_price = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_stock_price);
    const h_option_strike = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_option_strike);
    const h_option_years = try gpa.alloc(f32, opt_n);
    defer gpa.free(h_option_years);

    std.log.info("...allocating GPU memory for options.", .{});
    const d_call_result = try cuda.alloc(f32, opt_n);
    defer d_call_result.free();
    const d_put_result = try cuda.alloc(f32, opt_n);
    defer d_put_result.free();
    const d_stock_price = try cuda.alloc(f32, opt_n);
    defer d_stock_price.free();
    const d_option_strike = try cuda.alloc(f32, opt_n);
    defer d_option_strike.free();
    const d_option_years = try cuda.alloc(f32, opt_n);
    defer d_option_years.free();

    std.log.info("...generating input data in CPU mem.", .{});
    var prng: std.Random.DefaultPrng = .init(5347);
    const random = prng.random();

    // Generates the set of options.
    for (
        h_call_result_cpu,
        h_put_result_cpu,
        h_stock_price,
        h_option_strike,
        h_option_years,
    ) |*call, *put, *stock_price, *option_strike, *option_years| {
        call.* = 0.0;
        put.* = -1.0;
        stock_price.* = randFloat(random, 5.0, 30.0);
        option_strike.* = randFloat(random, 1.0, 100.0);
        option_years.* = randFloat(random, 0.25, 10.0);
    }

    std.log.info("...copying input data to GPU mem.", .{});
    try d_stock_price.copyFrom(h_stock_price);
    try d_option_strike.copyFrom(h_option_strike);
    try d_option_years.copyFrom(h_option_years);
    std.log.info("Data init done.", .{});

    std.log.info("Executing Black-Scholes GPU kernel ({d} iterations)...", .{num_iterations});
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    try cuda.synchronize();
    try start.record(.default);

    const work_group_size = 128;
    for (0..num_iterations) |_| {
        try black_scholes_gpu.launch(.{
            .grid = .{ .x = std.math.divCeil(u32, opt_n / 2, work_group_size) catch unreachable },
            .block = .{ .x = work_group_size },
        }, .{
            d_call_result,
            d_put_result,
            d_stock_price,
            d_option_strike,
            d_option_years,
            @as(f32, riskfree_rate),
            @as(f32, volatility_rate),
            @as(i32, opt_n),
        });
    }

    try stop.record(.default);
    try stop.synchronize();
    const gpu_time: f64 = try start.elapsed(stop) / num_iterations;

    // Both call and put are calculated.
    std.log.info("Options count             : {d}     ", .{2 * opt_n});
    std.log.info("BlackScholesGPU() time    : {d:.6} msec", .{gpu_time});
    std.log.info("Effective memory bandwidth: {d:.6} GB/s", .{
        (@as(f64, 5 * opt_n * @sizeOf(f32)) * 1e-9) / (gpu_time * 1e-3),
    });
    std.log.info("Gigaoptions per second    : {d:.6}     ", .{
        (@as(f64, 2 * opt_n) * 1e-9) / (gpu_time * 1e-3),
    });

    std.log.info("BlackScholes, Throughput = {d:.4} GOptions/s, Time = {d:.5} s, Size = {d} " ++
        "options, NumDevsUsed = {d}, Workgroup = {d}", .{
        (@as(f64, 2.0 * opt_n) * 1.0e-9) / (gpu_time * 1.0e-3),
        gpu_time * 1e-3,
        2 * opt_n,
        1,
        work_group_size,
    });

    std.log.info("Reading back GPU results...", .{});
    try d_call_result.copyTo(h_call_result_gpu);
    try d_put_result.copyTo(h_put_result_gpu);

    std.log.info("Checking the results...", .{});
    std.log.info("...running CPU calculations.", .{});
    blackScholesCpu(
        h_call_result_cpu,
        h_put_result_cpu,
        h_stock_price,
        h_option_strike,
        h_option_years,
        riskfree_rate,
        volatility_rate,
    );

    std.log.info("Comparing the results...", .{});
    // Calculates the maximum absolute difference and the L1 distance between the CPU and GPU
    // results.
    var sum_delta: f64 = 0;
    var sum_ref: f64 = 0;
    var max_delta: f64 = 0;

    for (h_call_result_cpu, h_call_result_gpu) |cpu, gpu| {
        const delta = @abs(@as(f64, cpu - gpu));
        max_delta = @max(max_delta, delta);
        sum_delta += delta;
        sum_ref += @abs(@as(f64, cpu));
    }

    const l1_norm = sum_delta / sum_ref;
    std.log.info("L1 norm: {E}", .{l1_norm});
    std.log.info("Max absolute error: {E}", .{max_delta});

    std.log.info("Shutting down...", .{});
    std.log.info("...releasing GPU memory.", .{});
    std.log.info("...releasing CPU memory.", .{});
    std.log.info("Shutdown done.", .{});

    std.log.info("[BlackScholes] - Test Summary", .{});

    if (l1_norm > 1e-6) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }

    std.log.info("NOTE: The CUDA Samples are not meant for performance measurements. " ++
        "Results may vary when GPU Boost is enabled.", .{});
    std.log.info("Test passed", .{});
}
