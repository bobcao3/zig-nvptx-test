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

//! Black-Scholes formula for the fair call and put prices of European options.

const std = @import("std");

const Float2 = @Vector(2, f32);

inline fn expApprox(x: f32) f32 {
    return std.nvptx.ex2Approx(x * std.math.log2e);
}

inline fn logApprox(x: f32) f32 {
    return std.nvptx.lg2Approx(x) * std.math.ln2;
}

/// Polynomial approximation of the cumulative normal distribution function.
inline fn cndGpu(d: f32) f32 {
    const a1: f32 = 0.31938153;
    const a2: f32 = -0.356563782;
    const a3: f32 = 1.781477937;
    const a4: f32 = -1.821255978;
    const a5: f32 = 1.330274429;
    const rsqrt_2pi: f32 = 0.39894228040143267793994605993438;

    const k = std.nvptx.divApprox(1.0, 1.0 + 0.2316419 * @abs(d));

    const polynomial = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))));
    var cnd = rsqrt_2pi * expApprox(-0.5 * d * d) * polynomial;

    if (d > 0) cnd = 1.0 - cnd;

    return cnd;
}

const Prices = struct { call: f32, put: f32 };

/// Black-Scholes formula for both call and put. `s` is the stock price, `x` the option strike,
/// `t` the option years, `r` the riskless rate and `v` the volatility rate.
inline fn blackScholesBodyGpu(s: f32, x: f32, t: f32, r: f32, v: f32) Prices {
    const sqrt_t = std.nvptx.divApprox(1.0, std.nvptx.rsqrtApprox(t));
    const d1 = std.nvptx.divApprox(logApprox(s / x) + (r + 0.5 * v * v) * t, v * sqrt_t);
    const d2 = d1 - v * sqrt_t;

    const cnd_d1 = cndGpu(d1);
    const cnd_d2 = cndGpu(d2);

    // Calculates call and put simultaneously.
    const exp_rt = expApprox(-r * t);
    return .{
        .call = s * cnd_d1 - x * exp_rt * cnd_d2,
        .put = x * exp_rt * (1.0 - cnd_d2) - s * (1.0 - cnd_d1),
    };
}

/// Processes an array of `opt_n` options. Each work item calculates two options to increase the
/// instruction level parallelism.
export fn blackScholesGpu(
    d_call_result: [*]align(8) addrspace(.global) Float2,
    d_put_result: [*]align(8) addrspace(.global) Float2,
    d_stock_price: [*]align(8) addrspace(.global) const Float2,
    d_option_strike: [*]align(8) addrspace(.global) const Float2,
    d_option_years: [*]align(8) addrspace(.global) const Float2,
    riskfree: f32,
    volatility: f32,
    opt_n: i32,
) callconv(.kernel) void {
    const opt = @workGroupSize(0) * @workGroupId(0) + @workItemId(0);

    if (opt < @divTrunc(opt_n, 2)) {
        const first = blackScholesBodyGpu(
            d_stock_price[opt][0],
            d_option_strike[opt][0],
            d_option_years[opt][0],
            riskfree,
            volatility,
        );
        const second = blackScholesBodyGpu(
            d_stock_price[opt][1],
            d_option_strike[opt][1],
            d_option_years[opt][1],
            riskfree,
            volatility,
        );
        d_call_result[opt] = .{ first.call, second.call };
        d_put_result[opt] = .{ first.put, second.put };
    }
}
