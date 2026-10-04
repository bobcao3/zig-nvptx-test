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

//! This sample evaluates the fair call price for a given set of European options under the
//! binomial model.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("binomialOptions_common.zig");

const Real = common.Real;
const OptionData = common.OptionData;
const num_steps = common.num_steps;
const max_options = common.max_options;

fn exp(x: Real) Real {
    return @floatCast(@exp(@as(f64, x)));
}

/// Polynomial approximation of the cumulative normal distribution function.
fn cnd(d: Real) Real {
    const a1: Real = 0.31938153;
    const a2: Real = -0.356563782;
    const a3: Real = 1.781477937;
    const a4: Real = -1.821255978;
    const a5: Real = 1.330274429;
    const rsqrt_2pi: Real = 0.39894228040143267793994605993438;

    const k: Real = @floatCast(1.0 / (1.0 + 0.2316419 * @as(f64, @abs(d))));

    const polynomial = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))));
    var result = rsqrt_2pi * @as(Real, @floatCast(@exp(-0.5 * @as(f64, d) * d))) * polynomial;

    if (d > 0) result = 1.0 - result;

    return result;
}

/// The Black-Scholes formula, which validates the results of the binomial tree.
fn blackScholesCall(option_data: OptionData) Real {
    const s = option_data.s;
    const x = option_data.x;
    const t = option_data.t;
    const r = option_data.r;
    const v = option_data.v;

    const sqrt_t: Real = @floatCast(@sqrt(@as(f64, t)));
    const d1: Real = @floatCast((@log(@as(f64, s / x)) + (r + 0.5 * v * v) * t) / (v * sqrt_t));
    const d2 = d1 - v * sqrt_t;
    const cnd_d1 = cnd(d1);
    const cnd_d2 = cnd(d2);

    const exp_rt = exp(-r * t);
    return s * cnd_d1 - x * exp_rt * cnd_d2;
}

fn expiryCallValue(s: Real, x: Real, v_dt: Real, i: usize) Real {
    const steps: Real = @floatFromInt(@as(i32, @intCast(2 * i)) - num_steps);
    const d = s * exp(v_dt * steps) - x;
    return if (d > 0) d else 0;
}

/// Processes a single option. The CPU code is for correctness testing only and not for
/// benchmarking.
fn binomialOptionsCpu(option_data: OptionData) Real {
    var call: [num_steps + 1]Real = undefined;

    const s = option_data.s;
    const x = option_data.x;
    const t = option_data.t;
    const r = option_data.r;
    const v = option_data.v;

    const dt = t / num_steps;
    const v_dt = v * @as(Real, @floatCast(@sqrt(@as(f64, dt))));
    const r_dt = r * dt;
    // Per-step interest and discount factors.
    const interest_factor = exp(r_dt);
    const discount_factor = exp(-r_dt);
    // Values and pseudoprobabilities of upward and downward moves.
    const u = exp(v_dt);
    const d = exp(-v_dt);
    const pu = (interest_factor - d) / (u - d);
    const pd = 1.0 - pu;
    const pu_by_df = pu * discount_factor;
    const pd_by_df = pd * discount_factor;

    // The value of a call option at the end of the period is S(T) - X if S(T) is greater than X,
    // and zero otherwise.
    for (&call, 0..) |*value, i| value.* = expiryCallValue(s, x, v_dt, i);

    // Walks backwards up the binomial tree.
    var i: usize = num_steps;
    while (i > 0) : (i -= 1) {
        for (0..i) |j| call[j] = pu_by_df * call[j + 1] + pd_by_df * call[j];
    }

    return call[0];
}

const Gpu = struct {
    binomial_options_kernel: cuda.Function,
    d_option_data: cuda.Buffer(common.PreprocessedOptionData),
    d_call_value: cuda.Buffer(Real),

    fn init(module: cuda.Module) !Gpu {
        return .{
            .binomial_options_kernel = try module.function("binomialOptionsKernel"),
            .d_option_data = try module.global(common.PreprocessedOptionData, "d_option_data"),
            .d_call_value = try module.global(Real, "d_call_value"),
        };
    }

    /// Writes the call values of `option_data` into `call_value`.
    fn binomialOptions(gpu: Gpu, call_value: []Real, option_data: []const OptionData) !void {
        var h_option_data: [max_options]common.PreprocessedOptionData = undefined;

        for (option_data, h_option_data[0..option_data.len]) |option, *preprocessed| {
            const dt = option.t / num_steps;
            const v_dt = option.v * @as(Real, @floatCast(@sqrt(@as(f64, dt))));
            const r_dt = option.r * dt;
            // Per-step interest and discount factors.
            const interest_factor = exp(r_dt);
            const discount_factor = exp(-r_dt);
            // Values and pseudoprobabilities of upward and downward moves.
            const u = exp(v_dt);
            const d = exp(-v_dt);
            const pu = (interest_factor - d) / (u - d);
            const pd = 1.0 - pu;

            preprocessed.* = .{
                .s = option.s,
                .x = option.x,
                .v_dt = v_dt,
                .pu_by_df = pu * discount_factor,
                .pd_by_df = pd * discount_factor,
            };
        }

        try gpu.d_option_data.copyFrom(h_option_data[0..option_data.len]);
        try gpu.binomial_options_kernel.launch(.{
            .grid = .{ .x = @intCast(option_data.len) },
            .block = .{ .x = 128 },
        }, .{});
        try gpu.d_call_value.copyTo(call_value);
    }
};

/// Returns a uniformly distributed random number in the range from `low` to `high`.
fn randData(random: std.Random, low: Real, high: Real) Real {
    const t = random.float(Real);
    return (1.0 - t) * low + t * high;
}

pub fn main() !void {
    std.log.info("[binomialOptions] - Starting...", .{});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("binomialOptions_kernel.ptx"));
    defer module.unload();
    const gpu: Gpu = try .init(module);

    const opt_n = max_options;

    var option_data: [max_options]OptionData = undefined;
    var call_value_bs: [max_options]Real = undefined;
    var call_value_gpu: [max_options]Real = undefined;
    var call_value_cpu: [max_options]Real = undefined;

    std.log.info("Generating input data...", .{});
    var prng: std.Random.DefaultPrng = .init(123);
    const random = prng.random();

    for (&option_data, &call_value_bs) |*option, *call_value| {
        option.* = .{
            .s = randData(random, 5.0, 30.0),
            .x = randData(random, 1.0, 100.0),
            .t = randData(random, 0.25, 10.0),
            .r = 0.06,
            .v = 0.10,
        };
        call_value.* = blackScholesCall(option.*);
    }

    std.log.info("Running GPU binomial tree...", .{});
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    try cuda.synchronize();
    try start.record(.default);

    try gpu.binomialOptions(&call_value_gpu, &option_data);

    try cuda.synchronize();
    try stop.record(.default);
    try stop.synchronize();
    const gpu_time = try start.elapsed(stop);
    std.log.info("Options count            : {d}     ", .{opt_n});
    std.log.info("Time steps               : {d}     ", .{num_steps});
    std.log.info("binomialOptionsGPU() time: {d:.6} msec", .{gpu_time});
    std.log.info("Options per second       : {d:.6}     ", .{opt_n / (gpu_time * 0.001)});

    std.log.info("Running CPU binomial tree...", .{});

    for (option_data, &call_value_cpu) |option, *call_value| {
        call_value.* = binomialOptionsCpu(option);
    }

    std.log.info("Comparing the results...", .{});
    var sum_delta: Real = 0;
    var sum_ref: Real = 0;
    std.log.info("GPU binomial vs. Black-Scholes", .{});

    for (call_value_bs, call_value_gpu) |reference, value| {
        sum_delta += @abs(reference - value);
        sum_ref += @abs(reference);
    }

    if (sum_ref > 1e-5) {
        std.log.info("L1 norm: {E}", .{@as(f64, sum_delta / sum_ref)});
    } else {
        std.log.info("Avg. diff: {E}", .{@as(f64, sum_delta / opt_n)});
    }

    std.log.info("CPU binomial vs. Black-Scholes", .{});
    sum_delta = 0;
    sum_ref = 0;

    for (call_value_bs, call_value_cpu) |reference, value| {
        sum_delta += @abs(reference - value);
        sum_ref += @abs(reference);
    }

    if (sum_ref > 1e-5) {
        std.log.info("L1 norm: {E}", .{sum_delta / sum_ref});
    } else {
        std.log.info("Avg. diff: {E}", .{@as(f64, sum_delta / opt_n)});
    }

    std.log.info("CPU binomial vs. GPU binomial", .{});
    sum_delta = 0;
    sum_ref = 0;

    for (call_value_gpu, call_value_cpu) |gpu_value, cpu_value| {
        sum_delta += @abs(gpu_value - cpu_value);
        sum_ref += cpu_value;
    }

    var error_val: Real = 0;
    if (sum_ref > 1e-5) {
        error_val = sum_delta / sum_ref;
        std.log.info("L1 norm: {E}", .{error_val});
    } else {
        std.log.info("Avg. diff: {E}", .{@as(f64, sum_delta / opt_n)});
    }

    std.log.info("Shutting down...", .{});

    std.log.info("NOTE: The CUDA Samples are not meant for performance measurements. " ++
        "Results may vary when GPU Boost is enabled.", .{});

    if (error_val > 5e-4) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }

    std.log.info("Test passed", .{});
}
