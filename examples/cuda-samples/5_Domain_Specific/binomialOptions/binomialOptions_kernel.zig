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

//! Binomial option pricing. Each work group evaluates the binomial tree of one option.

const std = @import("std");
const common = @import("binomialOptions_common.zig");

const Real = common.Real;
const num_steps = common.num_steps;

export var d_option_data: [common.max_options]common.PreprocessedOptionData addrspace(.global) =
    undefined;
export var d_call_value: [common.max_options]Real addrspace(.global) = undefined;

inline fn expiryCallValue(s: Real, x: Real, v_dt: Real, i: u32) Real {
    const steps: Real = @floatFromInt(i);
    const exponent = v_dt * (2.0 * steps - num_steps);
    const d = s * std.nvptx.ex2Approx(exponent * std.math.log2e) - x;
    return if (d > 0.0) d else 0.0;
}

const work_group_size = 128;
const elems_per_thread = num_steps / work_group_size;

comptime {
    if (num_steps % work_group_size != 0) @compileError("Bad constants");
}

var call_exchange: [work_group_size + 1]Real addrspace(.shared) = undefined;

export fn binomialOptionsKernel() callconv(.kernel) void {
    const tid = @workItemId(0);
    const option = d_option_data[@workGroupId(0)];
    const s = option.s;
    const x = option.x;
    const v_dt = option.v_dt;
    const pu_by_df = option.pu_by_df;
    const pd_by_df = option.pd_by_df;

    var call: [elems_per_thread + 1]Real = undefined;
    inline for (0..elems_per_thread) |i| {
        call[i] = expiryCallValue(s, x, v_dt, tid * elems_per_thread + @as(u32, i));
    }

    if (tid == 0) call_exchange[work_group_size] = expiryCallValue(s, x, v_dt, num_steps);

    const final_it = @max(0, @as(i32, @intCast(tid * elems_per_thread)) - 1);

    var i: i32 = num_steps;
    while (i > 0) : (i -= 1) {
        call_exchange[tid] = call[0];
        std.nvptx.barrierSyncAligned(.work_group, null);
        call[elems_per_thread] = call_exchange[tid + 1];
        std.nvptx.barrierSyncAligned(.work_group, null);

        if (i > final_it) {
            inline for (0..elems_per_thread) |j| {
                call[j] = pu_by_df * call[j + 1] + pd_by_df * call[j];
            }
        }
    }

    if (tid == 0) {
        d_call_value[@workGroupId(0)] = call[0];
    }
}
