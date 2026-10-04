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

//! The sizes of the GEMM computation that the host program and the kernels share.

const std = @import("std");

pub const warp_size = std.nvptx.warp_size;

pub const m = 8;
pub const n = 8;
pub const k = 4;

pub const m_tiles = 1024;
pub const n_tiles = 1024;
pub const k_tiles = 1024;

pub const m_global = m * m_tiles;
pub const n_global = n * n_tiles;
pub const k_global = k * k_tiles;

pub const warps_per_block = 8;
pub const threads_per_block = warp_size * warps_per_block;

/// A work group that caches chunks of 8 tiles of A and B needs 36 KiB of shared memory. Chunks of
/// 16 tiles need 68 KiB.
pub const shared_memory_limit_64k = false;
pub const chunk_k = if (shared_memory_limit_64k) 8 else 16;

pub const chunk_line_bytes = chunk_k * k * @sizeOf(f64);
pub const warp_copy_bytes = warp_size * @sizeOf(@Vector(4, u32));
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

/// The number of `f64` elements by which each row of A and each column of B in shared memory
/// shifts, so that different rows and columns fall into different shared memory banks. Four
/// elements are the smallest shift that keeps each row and column aligned to 32 bytes, as the
/// `wmma` load instructions require.
pub const skew_double = 4;

pub const shared_memory_bytes = @max(
    @sizeOf(f64) * (block_col_tiles * m) * (chunk_k * k + skew_double) * 2,
    m * (block_row_warps * warp_row_tiles) * n * (block_col_warps * warp_col_tiles) * @sizeOf(f64),
);
