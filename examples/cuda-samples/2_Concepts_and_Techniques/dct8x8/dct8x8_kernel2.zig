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

//! Contains 2nd kernel implementations of DCT and IDCT routines, used in JPEG internal data
//! processing. Optimized device code.
//!
//! This code implements traditional approach to forward and inverse Discrete Cosine Transform to
//! blocks of image pixels (of 8x8 size), as in JPEG standard. The data processing is done using
//! floating point representation. The routine that performs quantization of coefficients can be
//! found in dct8x8_kernel_quantization.zig file.

const std = @import("std");
const common = @import("common.zig");

const block_size = common.block_size;

const c_a = 1.387039845322148;
const c_b = 1.306562964876377;
const c_c = 1.175875602419359;
const c_d = 0.785694958387102;
const c_e = 0.541196100146197;
const c_f = 0.275899379282943;

const c_norm = 0.3535533905932737;

const ker2_block_width = common.ker2_block_width;
const ker2_block_height = common.ker2_block_height;
const ker2_smemblock_stride = ker2_block_width + 1;

var block: [ker2_block_height * ker2_smemblock_stride]f32 addrspace(.shared) = undefined;

inline fn subroutineInplaceDctVector(vect0: [*]addrspace(.shared) f32, step: u32) void {
    const x07p = vect0[0 * step] + vect0[7 * step];
    const x16p = vect0[1 * step] + vect0[6 * step];
    const x25p = vect0[2 * step] + vect0[5 * step];
    const x34p = vect0[3 * step] + vect0[4 * step];

    const x07m = vect0[0 * step] - vect0[7 * step];
    const x61m = vect0[6 * step] - vect0[1 * step];
    const x25m = vect0[2 * step] - vect0[5 * step];
    const x43m = vect0[4 * step] - vect0[3 * step];

    const x07p34pp = x07p + x34p;
    const x07p34pm = x07p - x34p;
    const x16p25pp = x16p + x25p;
    const x16p25pm = x16p - x25p;

    vect0[0 * step] = c_norm * (x07p34pp + x16p25pp);
    vect0[2 * step] = c_norm * (c_b * x07p34pm + c_e * x16p25pm);
    vect0[4 * step] = c_norm * (x07p34pp - x16p25pp);
    vect0[6 * step] = c_norm * (c_e * x07p34pm - c_b * x16p25pm);

    vect0[1 * step] = c_norm * (c_a * x07m - c_c * x61m + c_d * x25m - c_f * x43m);
    vect0[3 * step] = c_norm * (c_c * x07m + c_f * x61m - c_a * x25m + c_d * x43m);
    vect0[5 * step] = c_norm * (c_d * x07m + c_a * x61m + c_f * x25m - c_c * x43m);
    vect0[7 * step] = c_norm * (c_f * x07m + c_d * x61m + c_c * x25m + c_a * x43m);
}

inline fn subroutineInplaceIdctVector(vect0: [*]addrspace(.shared) f32, step: u32) void {
    const y04p = vect0[0 * step] + vect0[4 * step];
    const y2b6ep = c_b * vect0[2 * step] + c_e * vect0[6 * step];

    const y04p2b6epp = y04p + y2b6ep;
    const y04p2b6epm = y04p - y2b6ep;
    const y7f1ap3c5dpp = c_f * vect0[7 * step] + c_a * vect0[1 * step] +
        c_c * vect0[3 * step] + c_d * vect0[5 * step];
    const y7a1fm3d5cmp = c_a * vect0[7 * step] - c_f * vect0[1 * step] +
        c_d * vect0[3 * step] - c_c * vect0[5 * step];

    const y04m = vect0[0 * step] - vect0[4 * step];
    const y2e6bm = c_e * vect0[2 * step] - c_b * vect0[6 * step];

    const y04m2e6bmp = y04m + y2e6bm;
    const y04m2e6bmm = y04m - y2e6bm;
    const y1c7dm3f5apm = c_c * vect0[1 * step] - c_d * vect0[7 * step] -
        c_f * vect0[3 * step] - c_a * vect0[5 * step];
    const y1d7cp3a5fmm = c_d * vect0[1 * step] + c_c * vect0[7 * step] -
        c_a * vect0[3 * step] + c_f * vect0[5 * step];

    vect0[0 * step] = c_norm * (y04p2b6epp + y7f1ap3c5dpp);
    vect0[7 * step] = c_norm * (y04p2b6epp - y7f1ap3c5dpp);
    vect0[4 * step] = c_norm * (y04p2b6epm + y7a1fm3d5cmp);
    vect0[3 * step] = c_norm * (y04p2b6epm - y7a1fm3d5cmp);

    vect0[1 * step] = c_norm * (y04m2e6bmp + y1c7dm3f5apm);
    vect0[5 * step] = c_norm * (y04m2e6bmm - y1d7cp3a5fmm);
    vect0[2 * step] = c_norm * (y04m2e6bmm + y1d7cp3a5fmm);
    vect0[6 * step] = c_norm * (y04m2e6bmp - y1c7dm3f5apm);
}

/// Performs 8x8 block-wise Forward Discrete Cosine Transform of the given image plane and outputs
/// result to the array of coefficients. 2nd implementation. This kernel is designed to process
/// image by blocks of blocks8x8 that utilizes maximum warps capacity, assuming that it is enough
/// of 8 threads per block8x8.
pub export fn cudaKernel2Dct(
    dst: [*]addrspace(.global) f32,
    src: [*]addrspace(.global) f32,
    img_stride: u32,
) callconv(.kernel) void {
    const offs_thread_in_row = @workItemId(1) * block_size + @workItemId(0);
    const offs_thread_in_col = @workItemId(2) * block_size;
    const block_row = @workGroupId(1) * ker2_block_height + offs_thread_in_col;
    const offset = common.fmul(block_row, img_stride) +
        @workGroupId(0) * ker2_block_width + offs_thread_in_row;
    const src_thread = src + offset;
    const dst_thread = dst + offset;
    const shared_block: [*]addrspace(.shared) f32 = &block;
    const bl_ptr = shared_block + offs_thread_in_col * ker2_smemblock_stride + offs_thread_in_row;

    inline for (0..block_size) |i| bl_ptr[i * ker2_smemblock_stride] = src_thread[i * img_stride];

    std.nvptx.barrierSyncAligned(.work_group, null);
    subroutineInplaceDctVector(shared_block + (offs_thread_in_col + @workItemId(0)) *
        ker2_smemblock_stride + offs_thread_in_row - @workItemId(0), 1);

    std.nvptx.barrierSyncAligned(.work_group, null);
    subroutineInplaceDctVector(bl_ptr, ker2_smemblock_stride);

    std.nvptx.barrierSyncAligned(.work_group, null);
    for (0..block_size) |i| dst_thread[i * img_stride] = bl_ptr[i * ker2_smemblock_stride];
}

/// Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given coefficients plane and
/// outputs result to the image. 2nd implementation. This kernel is designed to process image by
/// blocks of blocks8x8 that utilizes maximum warps capacity, assuming that it is enough of 8
/// threads per block8x8.
pub export fn cudaKernel2Idct(
    dst: [*]addrspace(.global) f32,
    src: [*]addrspace(.global) f32,
    img_stride: u32,
) callconv(.kernel) void {
    const offs_thread_in_row = @workItemId(1) * block_size + @workItemId(0);
    const offs_thread_in_col = @workItemId(2) * block_size;
    const block_row = @workGroupId(1) * ker2_block_height + offs_thread_in_col;
    const offset = common.fmul(block_row, img_stride) +
        @workGroupId(0) * ker2_block_width + offs_thread_in_row;
    const src_thread = src + offset;
    const dst_thread = dst + offset;
    const shared_block: [*]addrspace(.shared) f32 = &block;
    const bl_ptr = shared_block + offs_thread_in_col * ker2_smemblock_stride + offs_thread_in_row;

    inline for (0..block_size) |i| bl_ptr[i * ker2_smemblock_stride] = src_thread[i * img_stride];

    std.nvptx.barrierSyncAligned(.work_group, null);
    subroutineInplaceIdctVector(shared_block + (offs_thread_in_col + @workItemId(0)) *
        ker2_smemblock_stride + offs_thread_in_row - @workItemId(0), 1);

    std.nvptx.barrierSyncAligned(.work_group, null);
    subroutineInplaceIdctVector(bl_ptr, ker2_smemblock_stride);

    std.nvptx.barrierSyncAligned(.work_group, null);

    for (0..block_size) |i| dst_thread[i * img_stride] = bl_ptr[i * ker2_smemblock_stride];
}
