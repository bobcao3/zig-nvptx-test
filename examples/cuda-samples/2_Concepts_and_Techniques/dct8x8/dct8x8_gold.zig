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

//! Contains DCT, IDCT and quantization routines, used in JPEG internal data processing. Host
//! code.
//!
//! This sample implements forward and inverse Discrete Cosine Transform to blocks of image pixels
//! (of 8x8 size), as in JPEG standard. The data processing is done using floating point
//! representation. The routine that performs quantization of coefficients is also included.

const std = @import("std");
const common = @import("common.zig");
const bmp_util = @import("bmp_util.zig");

const block_size = common.block_size;
const Roi = bmp_util.Roi;

/// This unitary matrix performs DCT of rows of the matrix to the left.
const dct_v8_matrix = [common.block_size2]f32{
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

/// This unitary matrix performs DCT of columns of the matrix to the right.
const dct_v8_matrix_t = [common.block_size2]f32{
    0.3535533905932738,  0.3535533905932738,  0.3535533905932738,  0.3535533905932738,
    0.3535533905932738,  0.3535533905932738,  0.3535533905932738,  0.3535533905932738,
    0.4903926402016152,  0.4157348061512726,  0.2777851165098011,  0.0975451610080642,
    -0.0975451610080641, -0.2777851165098010, -0.4157348061512727, -0.4903926402016152,
    0.4619397662556434,  0.1913417161825449,  -0.1913417161825449, -0.4619397662556434,
    -0.4619397662556434, -0.1913417161825452, 0.1913417161825450,  0.4619397662556433,
    0.4157348061512726,  -0.0975451610080641, -0.4903926402016152, -0.2777851165098011,
    0.2777851165098009,  0.4903926402016153,  0.0975451610080640,  -0.4157348061512721,
    0.3535533905932738,  -0.3535533905932737, -0.3535533905932738, 0.3535533905932737,
    0.3535533905932738,  -0.3535533905932733, -0.3535533905932736, 0.3535533905932733,
    0.2777851165098011,  -0.4903926402016152, 0.0975451610080642,  0.4157348061512727,
    -0.4157348061512726, -0.0975451610080649, 0.4903926402016152,  -0.2777851165098008,
    0.1913417161825449,  -0.4619397662556434, 0.4619397662556433,  -0.1913417161825450,
    -0.1913417161825453, 0.4619397662556437,  -0.4619397662556435, 0.1913417161825431,
    0.0975451610080642,  -0.2777851165098011, 0.4157348061512727,  -0.4903926402016153,
    0.4903926402016152,  -0.4157348061512720, 0.2777851165098022,  -0.0975451610080625,
};

/// JPEG quality=0_of_12 quantization matrix.
const q_gold = [common.block_size2]f32{
    32, 33, 51, 81,
    66, 39, 34, 17,
    33, 36, 48, 47,
    28, 23, 12, 12,
    51, 48, 47, 28,
    23, 12, 12, 12,
    81, 47, 28, 23,
    12, 12, 12, 12,
    66, 28, 23, 12,
    12, 12, 12, 12,
    39, 23, 12, 12,
    12, 12, 12, 12,
    34, 12, 12, 12,
    12, 12, 12, 12,
    17, 12, 12, 12,
    12, 12, 12, 12,
};

/// Performs multiplication of two 8x8 matrices.
fn mult8x8(
    m1: []const f32,
    m1_stride: u32,
    m2: []const f32,
    m2_stride: u32,
    mres: []f32,
    mres_stride: u32,
) void {
    for (0..block_size) |i| {
        for (0..block_size) |j| {
            var accumul: f32 = 0;
            for (0..block_size) |k| {
                accumul += m1[i * m1_stride + k] * m2[k * m2_stride + j];
            }
            mres[i * mres_stride + j] = accumul;
        }
    }
}

/// Performs 8x8 block-wise Forward Discrete Cosine Transform of the given image plane and outputs
/// result to the plane of coefficients. 1st version.
pub fn computeDct8x8Gold1(f_src: []const f32, f_dst: []f32, stride: u32, size: Roi) void {
    var tmpblock: [common.block_size2]f32 = undefined;

    // DCT(A) = dct_v8_matrix_t * A * dct_v8_matrix
    var i: u32 = 0;
    while (i + block_size - 1 < size.height) : (i += block_size) {
        var j: u32 = 0;
        while (j + block_size - 1 < size.width) : (j += block_size) {
            const block_src = f_src[i * stride + j ..];
            const block_dst = f_dst[i * stride + j ..];
            mult8x8(&dct_v8_matrix_t, block_size, block_src, stride, &tmpblock, block_size);
            mult8x8(&tmpblock, block_size, &dct_v8_matrix, block_size, block_dst, stride);
        }
    }
}

/// Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given coefficients plane and
/// outputs result to the image plane. 1st version.
pub fn computeIdct8x8Gold1(f_src: []const f32, f_dst: []f32, stride: u32, size: Roi) void {
    var tmpblock: [common.block_size2]f32 = undefined;

    // IDCT(A) = dct_v8_matrix * A * dct_v8_matrix_t
    var i: u32 = 0;
    while (i + block_size - 1 < size.height) : (i += block_size) {
        var j: u32 = 0;
        while (j + block_size - 1 < size.width) : (j += block_size) {
            const block_src = f_src[i * stride + j ..];
            const block_dst = f_dst[i * stride + j ..];
            mult8x8(&dct_v8_matrix, block_size, block_src, stride, &tmpblock, block_size);
            mult8x8(&tmpblock, block_size, &dct_v8_matrix_t, block_size, block_dst, stride);
        }
    }
}

/// Performs in-place quantization of given coefficients plane using predefined quantization
/// matrices (float elements).
pub fn quantizeGoldFloat(f_src_dst: []f32, stride: u32, size: Roi) void {
    // Q_GOLD(A) = round(A ./ Q_GOLD) .* Q_GOLD;
    for (0..size.height) |i| {
        for (0..size.width) |j| {
            const qx = j % block_size;
            const qy = i % block_size;
            const quantized = bmp_util.roundF(f_src_dst[i * stride + j] /
                q_gold[(qy << common.block_size_log2) + qx]);
            f_src_dst[i * stride + j] = quantized * q_gold[(qy << common.block_size_log2) + qx];
        }
    }
}

// Used in forward and inverse DCT.
const c_a = 1.387039845322148;
const c_b = 1.306562964876377;
const c_c = 1.175875602419359;
const c_d = 0.785694958387102;
const c_e = 0.541196100146197;
const c_f = 0.275899379282943;

const c_norm = 0.3535533905932737;

/// Performs DCT of vector of 8 elements.
fn subroutineDctVector(first_in: []const f32, step_in: u32, first_out: []f32, step_out: u32) void {
    const x07p = first_in[0 * step_in] + first_in[7 * step_in];
    const x16p = first_in[1 * step_in] + first_in[6 * step_in];
    const x25p = first_in[2 * step_in] + first_in[5 * step_in];
    const x34p = first_in[3 * step_in] + first_in[4 * step_in];

    const x07m = first_in[0 * step_in] - first_in[7 * step_in];
    const x61m = first_in[6 * step_in] - first_in[1 * step_in];
    const x25m = first_in[2 * step_in] - first_in[5 * step_in];
    const x43m = first_in[4 * step_in] - first_in[3 * step_in];

    const x07p34pp = x07p + x34p;
    const x07p34pm = x07p - x34p;
    const x16p25pp = x16p + x25p;
    const x16p25pm = x16p - x25p;

    first_out[0 * step_out] = c_norm * (x07p34pp + x16p25pp);
    first_out[2 * step_out] = c_norm * (c_b * x07p34pm + c_e * x16p25pm);
    first_out[4 * step_out] = c_norm * (x07p34pp - x16p25pp);
    first_out[6 * step_out] = c_norm * (c_e * x07p34pm - c_b * x16p25pm);

    first_out[1 * step_out] = c_norm * (c_a * x07m - c_c * x61m + c_d * x25m - c_f * x43m);
    first_out[3 * step_out] = c_norm * (c_c * x07m + c_f * x61m - c_a * x25m + c_d * x43m);
    first_out[5 * step_out] = c_norm * (c_d * x07m + c_a * x61m + c_f * x25m - c_c * x43m);
    first_out[7 * step_out] = c_norm * (c_f * x07m + c_d * x61m + c_c * x25m + c_a * x43m);
}

/// Performs IDCT of vector of 8 elements.
fn subroutineIdctVector(first_in: []const f32, step_in: u32, first_out: []f32, step_out: u32) void {
    const y04p = first_in[0 * step_in] + first_in[4 * step_in];
    const y2b6ep = c_b * first_in[2 * step_in] + c_e * first_in[6 * step_in];

    const y04p2b6epp = y04p + y2b6ep;
    const y04p2b6epm = y04p - y2b6ep;
    const y7f1ap3c5dpp =
        c_f * first_in[7 * step_in] + c_a * first_in[1 * step_in] +
        c_c * first_in[3 * step_in] + c_d * first_in[5 * step_in];
    const y7a1fm3d5cmp =
        c_a * first_in[7 * step_in] - c_f * first_in[1 * step_in] +
        c_d * first_in[3 * step_in] - c_c * first_in[5 * step_in];

    const y04m = first_in[0 * step_in] - first_in[4 * step_in];
    const y2e6bm = c_e * first_in[2 * step_in] - c_b * first_in[6 * step_in];

    const y04m2e6bmp = y04m + y2e6bm;
    const y04m2e6bmm = y04m - y2e6bm;
    const y1c7dm3f5apm =
        c_c * first_in[1 * step_in] - c_d * first_in[7 * step_in] -
        c_f * first_in[3 * step_in] - c_a * first_in[5 * step_in];
    const y1d7cp3a5fmm =
        c_d * first_in[1 * step_in] + c_c * first_in[7 * step_in] -
        c_a * first_in[3 * step_in] + c_f * first_in[5 * step_in];

    first_out[0 * step_out] = c_norm * (y04p2b6epp + y7f1ap3c5dpp);
    first_out[7 * step_out] = c_norm * (y04p2b6epp - y7f1ap3c5dpp);
    first_out[4 * step_out] = c_norm * (y04p2b6epm + y7a1fm3d5cmp);
    first_out[3 * step_out] = c_norm * (y04p2b6epm - y7a1fm3d5cmp);

    first_out[1 * step_out] = c_norm * (y04m2e6bmp + y1c7dm3f5apm);
    first_out[5 * step_out] = c_norm * (y04m2e6bmm - y1d7cp3a5fmm);
    first_out[2 * step_out] = c_norm * (y04m2e6bmm + y1d7cp3a5fmm);
    first_out[6 * step_out] = c_norm * (y04m2e6bmp - y1c7dm3f5apm);
}

/// Performs 8x8 block-wise Forward Discrete Cosine Transform of the given image plane and outputs
/// result to the plane of coefficients. 2nd version.
pub fn computeDct8x8Gold2(f_src: []const f32, f_dst: []f32, stride: u32, size: Roi) void {
    var i: u32 = 0;
    while (i + block_size - 1 < size.height) : (i += block_size) {
        var j: u32 = 0;
        while (j + block_size - 1 < size.width) : (j += block_size) {
            // process rows
            for (0..block_size) |k| {
                const offset = (i + k) * stride + j;
                subroutineDctVector(f_src[offset..], 1, f_dst[offset..], 1);
            }

            // process columns
            for (0..block_size) |k| {
                const offset = i * stride + (j + k);
                subroutineDctVector(f_dst[offset..], stride, f_dst[offset..], stride);
            }
        }
    }
}

/// Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given coefficients plane and
/// outputs result to the image plane. 2nd version.
pub fn computeIdct8x8Gold2(f_src: []const f32, f_dst: []f32, stride: u32, size: Roi) void {
    var i: u32 = 0;
    while (i + block_size - 1 < size.height) : (i += block_size) {
        var j: u32 = 0;
        while (j + block_size - 1 < size.width) : (j += block_size) {
            // process rows
            for (0..block_size) |k| {
                const offset = (i + k) * stride + j;
                subroutineIdctVector(f_src[offset..], 1, f_dst[offset..], 1);
            }

            // process columns
            for (0..block_size) |k| {
                const offset = i * stride + (j + k);
                subroutineIdctVector(f_dst[offset..], stride, f_dst[offset..], stride);
            }
        }
    }
}
