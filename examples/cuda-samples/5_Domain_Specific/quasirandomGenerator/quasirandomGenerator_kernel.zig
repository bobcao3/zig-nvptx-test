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

//! Niederreiter quasirandom number generation and the inverse cumulative normal distribution
//! function of Moro.

const std = @import("std");
const common = @import("quasirandomGenerator_common.zig");

const qrng_resolution = common.qrng_resolution;

export var c_table: [common.qrng_dimensions][qrng_resolution]u32 addrspace(.global) = undefined;

export fn quasirandomGeneratorKernel(
    d_output: [*]addrspace(.global) f32,
    seed: u32,
    n: u32,
) callconv(.kernel) void {
    const dim_base = &c_table[@workItemId(1)];
    const tid = @workGroupSize(0) * @workGroupId(0) + @workItemId(0);
    const thread_n = @workGroupSize(0) * std.nvptx.nctaid(0);

    var pos = tid;
    while (pos < n) : (pos += thread_n) {
        var result: u32 = 0;
        var data = seed +% pos;

        for (0..qrng_resolution) |bit| {
            if (data & 1 != 0) result ^= dim_base[bit];
            data >>= 1;
        }

        d_output[@workItemId(1) * n + pos] = @as(f32, @floatFromInt(result + 1)) * common.int_scale;
    }
}

inline fn logApprox(x: f32) f32 {
    return std.nvptx.lg2Approx(x) * std.math.ln2;
}

/// Moro's approximation of the inverse cumulative normal distribution function.
inline fn moroInvCndGpu(input: u32) f32 {
    const a1: f32 = 2.50662823884;
    const a2: f32 = -18.61500062529;
    const a3: f32 = 41.39119773534;
    const a4: f32 = -25.44106049637;
    const b1: f32 = -8.4735109309;
    const b2: f32 = 23.08336743743;
    const b3: f32 = -21.06224101826;
    const b4: f32 = 3.13082909833;
    const c1: f32 = 0.337475482272615;
    const c2: f32 = 0.976169019091719;
    const c3: f32 = 0.160797971491821;
    const c4: f32 = 2.76438810333863E-02;
    const c5: f32 = 3.8405729373609E-03;
    const c6: f32 = 3.951896511919E-04;
    const c7: f32 = 3.21767881768E-05;
    const c8: f32 = 2.888167364E-07;
    const c9: f32 = 3.960315187E-07;

    var x = input;
    var z: f32 = undefined;
    var negate = false;

    // Restricts the input to the bottom half of the input domain, so that the conversion to
    // floating point gives a value in the range (0,0.5]. The result is reflected later if the
    // input was in the top half of the domain.
    if (x >= 0x80000000) {
        x = 0xffffffff - x;
        negate = true;
    }

    // x is now in the range [0,0x7fffffff]. Converts it to floating point in (0,0.5].
    const x1: f32 = 1.0 / @as(f32, @floatFromInt(@as(u32, 0xffffffff)));
    const x2 = x1 / 2.0;
    const p1 = @as(f32, @floatFromInt(x)) * x1 + x2;
    // Converts to floating point in (-0.5,0].
    const p2 = p1 - 0.5;

    // Main body of the bell curve for |p| < 0.42.
    if (p2 > -0.42) {
        z = p2 * p2;
        const numerator = ((a4 * z + a3) * z + a2) * z + a1;
        const denominator = (((b4 * z + b3) * z + b2) * z + b1) * z + 1.0;
        z = p2 * numerator / denominator;
    } else {
        // Special case (Chebychev) for the tail.
        z = logApprox(-logApprox(p1));
        const upper = c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)));
        z = -(c1 + z * (c2 + z * (c3 + z * (c4 + z * upper))));
    }

    return if (negate) -z else z;
}

/// Transforms the input number sequence if it is supplied. Otherwise generates uniformly placed
/// samples on the fly.
export fn inverseCndKernel(
    d_output: [*]addrspace(.global) f32,
    d_input: ?[*]addrspace(.global) const u32,
    path_n: u32,
) callconv(.kernel) void {
    const distance = std.math.maxInt(u32) / (path_n + 1);
    const tid = @workGroupSize(0) * @workGroupId(0) + @workItemId(0);
    const thread_n = @workGroupSize(0) * std.nvptx.nctaid(0);

    if (d_input) |input| {
        var pos = tid;
        while (pos < path_n) : (pos += thread_n) {
            const d = input[pos];
            d_output[pos] = moroInvCndGpu(d);
        }
    } else {
        var pos = tid;
        while (pos < path_n) : (pos += thread_n) {
            const d = (pos + 1) * distance;
            d_output[pos] = moroInvCndGpu(d);
        }
    }
}
