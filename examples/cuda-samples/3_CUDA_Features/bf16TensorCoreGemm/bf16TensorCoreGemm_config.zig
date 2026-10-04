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

//! The dimensions of the matrices and of the tiles that the kernels and the host program share.

pub const warp_size = 32;

pub const m = 16;
pub const n = 16;
pub const k = 16;

pub const m_tiles = 512;
pub const n_tiles = 512;
pub const k_tiles = 512;

pub const m_global = m * m_tiles;
pub const n_global = n * n_tiles;
pub const k_global = k * k_tiles;

pub const warps_per_block = 8;
pub const threads_per_block = warp_size * warps_per_block;

pub const chunk_k = 8;

pub const chunk_line_bytes = chunk_k * k * @sizeOf(Bfloat16);
pub const warp_copy_bytes = warp_size * @sizeOf(Int4);
pub const chunk_copy_lines_per_warp = warp_copy_bytes / chunk_line_bytes;
pub const chunk_copy_line_lanes = warp_size / chunk_copy_lines_per_warp;

pub const block_row_warps = 2;
pub const block_col_warps = 4;

pub const warp_row_tiles = 4;
pub const warp_col_tiles = 2;

pub const block_row_tiles = warp_row_tiles * block_row_warps;
pub const block_col_tiles = warp_col_tiles * block_col_warps;

pub const global_mem_stride = n_global;

pub const shmem_stride = n * block_row_tiles;
pub const shmem_offset = n * warp_row_tiles;

/// The number of elements by which each row of A and each column of B in shared memory is
/// padded, so that different rows and columns map to different banks. The padded rows stay
/// 256-bit aligned, as `wmma.load` requires.
pub const skew_bf16 = 16;

/// The bit pattern of a bfloat16 number (1 sign bit, 8 exponent bits and 7 mantissa bits).
pub const Bfloat16 = u16;

pub const Int4 = @Vector(4, u32);
pub const Float4 = @Vector(4, f32);

/// The dynamic shared memory that the kernels with shared memory need: the larger of the tiles of
/// A and B and the tiles of C and D.
pub const shmem_size = @max(
    @sizeOf(Bfloat16) * (block_col_tiles * m) * (chunk_k * k + skew_bf16) * 2,
    m * (block_row_warps * warp_row_tiles) * n * (block_col_warps * warp_col_tiles) * @sizeOf(f32),
);
