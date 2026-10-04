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

//! Contains kernel implementations of DCT and IDCT routines for 16-bit integers, used in JPEG
//! internal data processing. Optimized device code.
//!
//! This code implements traditional approach to forward and inverse Discrete Cosine Transform to
//! blocks of image pixels (of 8x8 size), as in JPEG standard. The data processing is performed
//! using short data type. The routine that performs quantization of coefficients can be found in
//! dct8x8_kernel_quantization.zig file.

const std = @import("std");
const common = @import("common.zig");

const block_size = common.block_size;
const fmul = common.fmul;

const kers_block_width = common.kers_block_width;
const kers_block_height = common.kers_block_height;
const kers_smemblock_stride = kers_block_width + 2;
const kers_block_width_half = kers_block_width / 2;

const sin_1_4: i32 = 0x5A82;
const cos_1_4: i32 = 0x5A82;
const sin_1_8: i32 = 0x30FC;
const cos_1_8: i32 = 0x7642;

const osin_1_16: i32 = 0x063E;
const osin_3_16: i32 = 0x11C7;
const osin_5_16: i32 = 0x1A9B;
const osin_7_16: i32 = 0x1F63;

const ocos_1_16: i32 = 0x1F63;
const ocos_3_16: i32 = 0x1A9B;
const ocos_5_16: i32 = 0x11C7;
const ocos_7_16: i32 = 0x063E;

var block: [kers_block_height * kers_smemblock_stride]i16 align(4) addrspace(.shared) = undefined;

/// Converts fixed point value to short value.
inline fn unfixh(x: i32) i16 {
    return @truncate((x + 0x8000) >> 16);
}

/// Converts fixed point value to short value.
inline fn unfixo(x: i32) i32 {
    return (x + 0x1000) >> 13;
}

/// Performs in-place DCT of vector of 8 elements (used to access columns in shared memory).
inline fn shortInplaceDctColumn(src_dst: [*]addrspace(.shared) i16, stride: u32) void {
    const double_stride = stride << 1;

    const in0: i32 = src_dst[0 * stride];
    const in1: i32 = src_dst[1 * stride];
    const in2: i32 = src_dst[2 * stride];
    const in3: i32 = src_dst[3 * stride];
    const in4: i32 = src_dst[4 * stride];
    const in5: i32 = src_dst[5 * stride];
    const in6: i32 = src_dst[6 * stride];
    const in7: i32 = src_dst[7 * stride];

    const tmp0 = in7 + in0;
    const tmp1 = in6 + in1;
    const tmp2 = in5 + in2;
    const tmp3 = in4 + in3;
    var tmp4 = in3 - in4;
    const tmp5 = in2 - in5;
    const tmp6 = in1 - in6;
    var tmp7 = in0 - in7;

    const tmp10 = tmp3 + tmp0;
    const tmp11 = tmp2 + tmp1;
    const tmp12 = tmp1 - tmp2;
    const tmp13 = tmp0 - tmp3;

    const tmp16 = unfixo(fmul(tmp6 + tmp5, sin_1_4));
    const tmp15 = unfixo(fmul(tmp6 - tmp5, cos_1_4));

    tmp4 <<= 2;
    tmp7 <<= 2;

    const tmp14 = tmp4 + tmp15;
    const tmp25 = tmp4 - tmp15;
    const tmp26 = tmp7 - tmp16;
    const tmp17 = tmp7 + tmp16;

    src_dst[0 * double_stride] = unfixh(fmul(tmp10 + tmp11, sin_1_4));
    src_dst[1 * double_stride] = unfixh(fmul(tmp13, cos_1_8) + fmul(tmp12, sin_1_8));
    src_dst[2 * double_stride] = unfixh(fmul(tmp10 - tmp11, cos_1_4));
    src_dst[3 * double_stride] = unfixh(fmul(tmp13, sin_1_8) - fmul(tmp12, cos_1_8));

    const odd = src_dst + stride;
    odd[0 * double_stride] = unfixh(fmul(tmp17, ocos_1_16) + fmul(tmp14, osin_1_16));
    odd[1 * double_stride] = unfixh(fmul(tmp26, ocos_3_16) - fmul(tmp25, osin_3_16));
    odd[2 * double_stride] = unfixh(fmul(tmp26, ocos_5_16) + fmul(tmp25, osin_5_16));
    odd[3 * double_stride] = unfixh(fmul(tmp17, ocos_7_16) - fmul(tmp14, osin_7_16));
}

/// Performs in-place DCT of vector of 8 elements (used to access rows in shared memory).
inline fn shortInplaceDctRow(v8: [*]addrspace(.shared) u32) void {
    var sh0: [2]i16 = @bitCast(v8[0]);
    var sh1: [2]i16 = @bitCast(v8[1]);
    var sh2: [2]i16 = @bitCast(v8[2]);
    var sh3: [2]i16 = @bitCast(v8[3]);
    const in0: i32 = sh0[0];
    const in1: i32 = sh0[1];
    const in2: i32 = sh1[0];
    const in3: i32 = sh1[1];
    const in4: i32 = sh2[0];
    const in5: i32 = sh2[1];
    const in6: i32 = sh3[0];
    const in7: i32 = sh3[1];

    const tmp0 = in7 + in0;
    const tmp1 = in6 + in1;
    const tmp2 = in5 + in2;
    const tmp3 = in4 + in3;
    var tmp4 = in3 - in4;
    const tmp5 = in2 - in5;
    const tmp6 = in1 - in6;
    var tmp7 = in0 - in7;

    const tmp10 = tmp3 + tmp0;
    const tmp11 = tmp2 + tmp1;
    const tmp12 = tmp1 - tmp2;
    const tmp13 = tmp0 - tmp3;

    sh0[0] = unfixh(fmul(tmp10 + tmp11, sin_1_4));
    sh2[0] = unfixh(fmul(tmp10 - tmp11, cos_1_4));

    sh1[0] = unfixh(fmul(tmp13, cos_1_8) + fmul(tmp12, sin_1_8));
    sh3[0] = unfixh(fmul(tmp13, sin_1_8) - fmul(tmp12, cos_1_8));

    const tmp16 = unfixo(fmul(tmp6 + tmp5, sin_1_4));
    const tmp15 = unfixo(fmul(tmp6 - tmp5, cos_1_4));

    tmp4 <<= 2;
    tmp7 <<= 2;

    const tmp14 = tmp4 + tmp15;
    const tmp25 = tmp4 - tmp15;
    const tmp26 = tmp7 - tmp16;
    const tmp17 = tmp7 + tmp16;

    sh0[1] = unfixh(fmul(tmp17, ocos_1_16) + fmul(tmp14, osin_1_16));
    sh3[1] = unfixh(fmul(tmp17, ocos_7_16) - fmul(tmp14, osin_7_16));
    sh2[1] = unfixh(fmul(tmp26, ocos_5_16) + fmul(tmp25, osin_5_16));
    sh1[1] = unfixh(fmul(tmp26, ocos_3_16) - fmul(tmp25, osin_3_16));

    v8[0] = @bitCast(sh0);
    v8[1] = @bitCast(sh1);
    v8[2] = @bitCast(sh2);
    v8[3] = @bitCast(sh3);
}

/// Performs in-place IDCT of vector of 8 elements (used to access columns in shared memory).
inline fn shortInplaceIdctColumn(src_dst: [*]addrspace(.shared) i16, stride: u32) void {
    const in0: i32 = src_dst[0 * stride];
    var in1: i32 = src_dst[1 * stride];
    const in2: i32 = src_dst[2 * stride];
    const in3: i32 = src_dst[3 * stride];
    const in4: i32 = src_dst[4 * stride];
    const in5: i32 = src_dst[5 * stride];
    const in6: i32 = src_dst[6 * stride];
    var in7: i32 = src_dst[7 * stride];

    const tmp10 = fmul(in0 + in4, cos_1_4);
    const tmp11 = fmul(in0 - in4, cos_1_4);
    const tmp12 = fmul(in2, sin_1_8) - fmul(in6, cos_1_8);
    const tmp13 = fmul(in6, sin_1_8) + fmul(in2, cos_1_8);

    const tmp20 = tmp10 + tmp13;
    const tmp21 = tmp11 + tmp12;
    const tmp22 = tmp11 - tmp12;
    const tmp23 = tmp10 - tmp13;

    const tmp30 = unfixo(fmul(in3 + in5, cos_1_4));
    const tmp31 = unfixo(fmul(in3 - in5, cos_1_4));

    in1 <<= 2;
    in7 <<= 2;

    const tmp40 = in1 + tmp30;
    const tmp41 = in7 + tmp31;
    const tmp42 = in1 - tmp30;
    const tmp43 = in7 - tmp31;

    const tmp50 = fmul(tmp40, ocos_1_16) + fmul(tmp41, osin_1_16);
    const tmp51 = fmul(tmp40, osin_1_16) - fmul(tmp41, ocos_1_16);
    const tmp52 = fmul(tmp42, ocos_5_16) + fmul(tmp43, osin_5_16);
    const tmp53 = fmul(tmp42, osin_5_16) - fmul(tmp43, ocos_5_16);

    src_dst[0 * stride] = unfixh(tmp20 + tmp50);
    src_dst[1 * stride] = unfixh(tmp21 + tmp53);
    src_dst[2 * stride] = unfixh(tmp22 + tmp52);
    src_dst[3 * stride] = unfixh(tmp23 + tmp51);
    src_dst[4 * stride] = unfixh(tmp23 - tmp51);
    src_dst[5 * stride] = unfixh(tmp22 - tmp52);
    src_dst[6 * stride] = unfixh(tmp21 - tmp53);
    src_dst[7 * stride] = unfixh(tmp20 - tmp50);
}

/// Performs in-place IDCT of vector of 8 elements (used to access rows in shared memory).
inline fn shortInplaceIdctRow(v8: [*]addrspace(.shared) u32) void {
    var sh0: [2]i16 = @bitCast(v8[0]);
    var sh1: [2]i16 = @bitCast(v8[1]);
    var sh2: [2]i16 = @bitCast(v8[2]);
    var sh3: [2]i16 = @bitCast(v8[3]);
    const in0: i32 = sh0[0];
    var in1: i32 = sh0[1];
    const in2: i32 = sh1[0];
    const in3: i32 = sh1[1];
    const in4: i32 = sh2[0];
    const in5: i32 = sh2[1];
    const in6: i32 = sh3[0];
    var in7: i32 = sh3[1];

    const tmp10 = fmul(in0 + in4, cos_1_4);
    const tmp11 = fmul(in0 - in4, cos_1_4);
    const tmp12 = fmul(in2, sin_1_8) - fmul(in6, cos_1_8);
    const tmp13 = fmul(in6, sin_1_8) + fmul(in2, cos_1_8);

    const tmp20 = tmp10 + tmp13;
    const tmp21 = tmp11 + tmp12;
    const tmp22 = tmp11 - tmp12;
    const tmp23 = tmp10 - tmp13;

    const tmp30 = unfixo(fmul(in3 + in5, cos_1_4));
    const tmp31 = unfixo(fmul(in3 - in5, cos_1_4));

    in1 <<= 2;
    in7 <<= 2;

    const tmp40 = in1 + tmp30;
    const tmp41 = in7 + tmp31;
    const tmp42 = in1 - tmp30;
    const tmp43 = in7 - tmp31;

    const tmp50 = fmul(tmp40, ocos_1_16) + fmul(tmp41, osin_1_16);
    const tmp51 = fmul(tmp40, osin_1_16) - fmul(tmp41, ocos_1_16);
    const tmp52 = fmul(tmp42, ocos_5_16) + fmul(tmp43, osin_5_16);
    const tmp53 = fmul(tmp42, osin_5_16) - fmul(tmp43, ocos_5_16);

    sh0[0] = unfixh(tmp20 + tmp50);
    sh0[1] = unfixh(tmp21 + tmp53);
    sh1[0] = unfixh(tmp22 + tmp52);
    sh1[1] = unfixh(tmp23 + tmp51);
    sh2[0] = unfixh(tmp23 - tmp51);
    sh2[1] = unfixh(tmp22 - tmp52);
    sh3[0] = unfixh(tmp21 - tmp53);
    sh3[1] = unfixh(tmp20 - tmp50);

    v8[0] = @bitCast(sh0);
    v8[1] = @bitCast(sh1);
    v8[2] = @bitCast(sh2);
    v8[3] = @bitCast(sh3);
}

inline fn permutedThreadInRow(offs_thread_in_row: u32) u32 {
    return (offs_thread_in_row & 0xFFFFFFE0) |
        (((offs_thread_in_row << 1) | ((offs_thread_in_row >> 4) & 0x1)) & 0x1F);
}

inline fn loadBlock(
    bl_ptr: [*]addrspace(.shared) i16,
    src_dst: [*]addrspace(.global) i16,
    img_stride: u32,
    offs_thread_in_row: u32,
) void {
    if (offs_thread_in_row < kers_block_width_half) {
        const shared_words: [*]addrspace(.shared) u32 = @ptrCast(@alignCast(bl_ptr));
        const global_words: [*]addrspace(.global) u32 = @ptrCast(@alignCast(src_dst));
        inline for (0..block_size) |i| {
            shared_words[i * (kers_smemblock_stride / 2)] = global_words[i * (img_stride / 2)];
        }
    }
}

inline fn storeBlock(
    bl_ptr: [*]addrspace(.shared) i16,
    src_dst: [*]addrspace(.global) i16,
    img_stride: u32,
    offs_thread_in_row: u32,
) void {
    if (offs_thread_in_row < kers_block_width_half) {
        const shared_words: [*]addrspace(.shared) u32 = @ptrCast(@alignCast(bl_ptr));
        const global_words: [*]addrspace(.global) u32 = @ptrCast(@alignCast(src_dst));
        inline for (0..block_size) |i| {
            global_words[i * (img_stride / 2)] = shared_words[i * (kers_smemblock_stride / 2)];
        }
    }
}

/// Performs 8x8 block-wise Forward Discrete Cosine Transform of the given image plane and outputs
/// result to the array of coefficients. Short implementation. This kernel is designed to process
/// image by blocks of blocks8x8 that utilize maximum warps capacity, assuming that it is enough
/// of 8 threads per block8x8.
pub export fn cudaKernelShortDct(
    src_dst: [*]addrspace(.global) i16,
    img_stride: u32,
) callconv(.kernel) void {
    const shared_block: [*]addrspace(.shared) i16 = &block;
    const offs_thread_in_row = fmul(@workItemId(1), block_size) + @workItemId(0);
    const offs_thread_in_col = fmul(@workItemId(2), block_size);
    const offs_thr_row_permuted = permutedThreadInRow(offs_thread_in_row);

    const src_dst_thread = src_dst + ((@workGroupId(1) * kers_block_height + offs_thread_in_col) *
        img_stride + (@workGroupId(0) * kers_block_width + offs_thread_in_row * 2));
    const bl_ptr = shared_block +
        (offs_thread_in_col * kers_smemblock_stride + offs_thread_in_row * 2);

    loadBlock(bl_ptr, src_dst_thread, img_stride, offs_thread_in_row);

    std.nvptx.barrierSyncAligned(.work_group, null);
    shortInplaceDctColumn(shared_block + offs_thread_in_col * kers_smemblock_stride +
        offs_thr_row_permuted, kers_smemblock_stride);
    std.nvptx.barrierSyncAligned(.work_group, null);
    const row: [*]addrspace(.shared) u32 = @ptrCast(@alignCast(shared_block +
        offs_thread_in_row * kers_smemblock_stride + offs_thread_in_col));
    shortInplaceDctRow(row);
    std.nvptx.barrierSyncAligned(.work_group, null);

    storeBlock(bl_ptr, src_dst_thread, img_stride, offs_thread_in_row);
}

/// Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given image plane and outputs
/// result to the array of coefficients. Short implementation. This kernel is designed to process
/// image by blocks of blocks8x8 that utilize maximum warps capacity, assuming that it is enough
/// of 8 threads per block8x8.
pub export fn cudaKernelShortIdct(
    src_dst: [*]addrspace(.global) i16,
    img_stride: u32,
) callconv(.kernel) void {
    const shared_block: [*]addrspace(.shared) i16 = &block;
    const offs_thread_in_row = @workItemId(1) * block_size + @workItemId(0);
    const offs_thread_in_col = @workItemId(2) * block_size;
    const offs_thr_row_permuted = permutedThreadInRow(offs_thread_in_row);

    const src_dst_thread = src_dst + ((@workGroupId(1) * kers_block_height + offs_thread_in_col) *
        img_stride + (@workGroupId(0) * kers_block_width + offs_thread_in_row * 2));
    const bl_ptr = shared_block +
        (offs_thread_in_col * kers_smemblock_stride + offs_thread_in_row * 2);

    loadBlock(bl_ptr, src_dst_thread, img_stride, offs_thread_in_row);

    std.nvptx.barrierSyncAligned(.work_group, null);
    shortInplaceIdctColumn(shared_block + offs_thread_in_col * kers_smemblock_stride +
        offs_thr_row_permuted, kers_smemblock_stride);
    std.nvptx.barrierSyncAligned(.work_group, null);
    const row: [*]addrspace(.shared) u32 = @ptrCast(@alignCast(shared_block +
        offs_thread_in_row * kers_smemblock_stride + offs_thread_in_col));
    shortInplaceIdctRow(row);
    std.nvptx.barrierSyncAligned(.work_group, null);

    storeBlock(bl_ptr, src_dst_thread, img_stride, offs_thread_in_row);
}
