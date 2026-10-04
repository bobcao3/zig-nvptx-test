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

//! Contains 1st CUDA implementations of DCT, IDCT and quantization routines, used in JPEG
//! internal data processing. Device code.
//!
//! This code implements first CUDA versions of forward and inverse Discrete Cosine Transform to
//! blocks of image pixels (of 8x8 size), as in JPEG standard. The data processing is done using
//! floating point representation. The routine that performs quantization of coefficients can be
//! found in dct8x8_kernel_quantization.zig file.

const std = @import("std");
const common = @import("common.zig");

const block_size = common.block_size;
const block_size_log2 = common.block_size_log2;

/// This unitary matrix performs discrete cosine transform of rows of the matrix to the left.
var dct_v8_matrix: [common.block_size2]f32 addrspace(.global) = .{
    0.3535533905932738,  0.4903926402016152,  0.4619397662556434,  0.4157348061512726,
    0.3535533905932738,  0.2777851165098011,  0.1913417161825449,  0.0975451610080642,
    0.3535533905932738,  0.4157348061512726,  0.1913417161825449,  -0.0975451610080641,
    -0.3535533905932737, -0.4903926402016152, -0.4619397662556434, -0.2777851165098011,
    0.3535533905932738,  0.2777851165098011,  -0.1913417161825449, -0.4903926402016152,
    -0.3535533905932738, 0.0975451610080642,  0.4619397662556433,  0.4157348061512727,
    0.3535533905932738,  0.0975451610080642,  -0.4619397662556434, -0.2777851165098011,
    0.3535533905932737,  0.4157348061512727,  -0.1913417161825450, -0.4903926402016153,
    0.3535533905932738,  -0.0975451610080641, -0.4619397662556434, 0.2777851165098009,
    0.3535533905932738,  -0.4157348061512726, -0.1913417161825453, 0.4903926402016152,
    0.3535533905932738,  -0.2777851165098010, -0.1913417161825452, 0.4903926402016153,
    -0.3535533905932733, -0.0975451610080649, 0.4619397662556437,  -0.4157348061512720,
    0.3535533905932738,  -0.4157348061512727, 0.1913417161825450,  0.0975451610080640,
    -0.3535533905932736, 0.4903926402016152,  -0.4619397662556435, 0.2777851165098022,
    0.3535533905932738,  -0.4903926402016152, 0.4619397662556433,  -0.4157348061512721,
    0.3535533905932733,  -0.2777851165098008, 0.1913417161825431,  -0.0975451610080625,
};

var cur_block_local1: [common.block_size2]f32 addrspace(.shared) = undefined;
var cur_block_local2: [common.block_size2]f32 addrspace(.shared) = undefined;

/// Performs 1st implementation of 8x8 block-wise Forward Discrete Cosine Transform of the given
/// image plane and outputs result to the array of coefficients.
pub export fn cudaKernel1Dct(
    dst: [*]addrspace(.global) f32,
    img_width: u32,
    offset_x_blocks: u32,
    offset_y_blocks: u32,
    tex_src: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const bx = @workGroupId(0) + offset_x_blocks;
    const by = @workGroupId(1) + offset_y_blocks;

    const tx = @workItemId(0);
    const ty = @workItemId(1);

    const tex_x = @as(f32, @floatFromInt((bx << block_size_log2) + tx)) + 0.5;
    const tex_y = @as(f32, @floatFromInt((by << block_size_log2) + ty)) + 0.5;

    const texel = tex_src.tex(.@"2d", f32, f32, .{ tex_x, tex_y });
    cur_block_local1[(ty << block_size_log2) + tx] = texel[0];

    std.nvptx.barrierSyncAligned(.work_group, null);

    var curelem: f32 = 0;
    var dct_v8_matrix_index = 0 * block_size + ty;
    var cur_block_local1_index = 0 * block_size + tx;
    inline for (0..block_size) |_| {
        curelem += dct_v8_matrix[dct_v8_matrix_index] * cur_block_local1[cur_block_local1_index];
        dct_v8_matrix_index += block_size;
        cur_block_local1_index += block_size;
    }

    cur_block_local2[(ty << block_size_log2) + tx] = curelem;

    std.nvptx.barrierSyncAligned(.work_group, null);

    curelem = 0;
    var cur_block_local2_index = (ty << block_size_log2) + 0;
    dct_v8_matrix_index = 0 * block_size + tx;
    inline for (0..block_size) |_| {
        curelem += cur_block_local2[cur_block_local2_index] * dct_v8_matrix[dct_v8_matrix_index];
        cur_block_local2_index += 1;
        dct_v8_matrix_index += block_size;
    }

    cur_block_local1[(ty << block_size_log2) + tx] = curelem;

    std.nvptx.barrierSyncAligned(.work_group, null);

    dst[common.fmul((by << block_size_log2) + ty, img_width) + (bx << block_size_log2) + tx] =
        cur_block_local1[(ty << block_size_log2) + tx];
}

/// Performs 1st implementation of 8x8 block-wise Inverse Discrete Cosine Transform of the given
/// DCT coefficients plane and outputs result to the image array.
pub export fn cudaKernel1Idct(
    dst: [*]addrspace(.global) f32,
    img_width: u32,
    offset_x_blocks: u32,
    offset_y_blocks: u32,
    tex_src: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const bx = @workGroupId(0) + offset_x_blocks;
    const by = @workGroupId(1) + offset_y_blocks;

    const tx = @workItemId(0);
    const ty = @workItemId(1);

    const tex_x = @as(f32, @floatFromInt((bx << block_size_log2) + tx)) + 0.5;
    const tex_y = @as(f32, @floatFromInt((by << block_size_log2) + ty)) + 0.5;

    const texel = tex_src.tex(.@"2d", f32, f32, .{ tex_x, tex_y });
    cur_block_local1[(ty << block_size_log2) + tx] = texel[0];

    std.nvptx.barrierSyncAligned(.work_group, null);

    var curelem: f32 = 0;
    var dct_v8_matrix_index = (ty << block_size_log2) + 0;
    var cur_block_local1_index = 0 * block_size + tx;
    inline for (0..block_size) |_| {
        curelem += dct_v8_matrix[dct_v8_matrix_index] * cur_block_local1[cur_block_local1_index];
        dct_v8_matrix_index += 1;
        cur_block_local1_index += block_size;
    }

    cur_block_local2[(ty << block_size_log2) + tx] = curelem;

    std.nvptx.barrierSyncAligned(.work_group, null);

    curelem = 0;
    var cur_block_local2_index = (ty << block_size_log2) + 0;
    dct_v8_matrix_index = (tx << block_size_log2) + 0;
    inline for (0..block_size) |_| {
        curelem += cur_block_local2[cur_block_local2_index] * dct_v8_matrix[dct_v8_matrix_index];
        cur_block_local2_index += 1;
        dct_v8_matrix_index += 1;
    }

    cur_block_local1[(ty << block_size_log2) + tx] = curelem;

    std.nvptx.barrierSyncAligned(.work_group, null);

    dst[common.fmul((by << block_size_log2) + ty, img_width) + (bx << block_size_log2) + tx] =
        cur_block_local1[(ty << block_size_log2) + tx];
}
