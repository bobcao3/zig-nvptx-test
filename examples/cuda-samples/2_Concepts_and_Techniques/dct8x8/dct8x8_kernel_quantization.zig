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

//! Contains unoptimized quantization routines. Device code.
//!
//! This code implements CUDA versions of quantization of Discrete Cosine Transform coefficients
//! with 8x8 blocks for float and short arrays.

const std = @import("std");
const common = @import("common.zig");

const block_size = common.block_size;

/// JPEG quality=0_of_12 quantization matrix.
var q: [common.block_size2]i16 addrspace(.global) = .{
    32, 33, 51, 81, 66, 39, 34, 17, 33, 36, 48, 47, 28, 23, 12, 12, 51, 48, 47, 28, 23, 12,
    12, 12, 81, 47, 28, 23, 12, 12, 12, 12, 66, 28, 23, 12, 12, 12, 12, 12, 39, 23, 12, 12,
    12, 12, 12, 12, 34, 12, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12,
};

/// Performs in-place quantization of given DCT coefficients plane using predefined quantization
/// matrices (for floats plane). Unoptimized.
pub export fn cudaKernelQuantizationFloat(
    src_dst: [*]addrspace(.global) f32,
    stride: u32,
) callconv(.kernel) void {
    const bx = @workGroupId(0);
    const by = @workGroupId(1);

    const tx = @workItemId(0);
    const ty = @workItemId(1);

    var cur_coef = src_dst[(by * block_size + ty) * stride + (bx * block_size + tx)];
    const cur_quant: f32 = @floatFromInt(q[ty * block_size + tx]);

    const quantized = @round(cur_coef / cur_quant);
    cur_coef = quantized * cur_quant;

    src_dst[(by * block_size + ty) * stride + (bx * block_size + tx)] = cur_coef;
}

/// Performs in-place quantization of given DCT coefficients plane using predefined quantization
/// matrices (for shorts plane). Unoptimized.
pub export fn cudaKernelQuantizationShort(
    src_dst: [*]addrspace(.global) i16,
    stride: u32,
) callconv(.kernel) void {
    const bx = @workGroupId(0);
    const by = @workGroupId(1);

    const tx = @workItemId(0);
    const ty = @workItemId(1);

    var cur_coef = src_dst[(by * block_size + ty) * stride + (bx * block_size + tx)];
    const cur_quant = q[ty * block_size + tx];

    if (cur_coef < 0) {
        cur_coef = -%cur_coef;
        cur_coef +%= cur_quant >> 1;
        cur_coef = @divTrunc(cur_coef, cur_quant);
        cur_coef = -%cur_coef;
    } else {
        cur_coef +%= cur_quant >> 1;
        cur_coef = @divTrunc(cur_coef, cur_quant);
    }

    std.nvptx.barrierSyncAligned(.work_group, null);

    cur_coef = cur_coef *% cur_quant;

    src_dst[(by * block_size + ty) * stride + (bx * block_size + tx)] = cur_coef;
}
