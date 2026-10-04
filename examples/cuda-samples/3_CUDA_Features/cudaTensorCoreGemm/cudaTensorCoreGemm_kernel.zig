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

//! The kernels of a GEMM computation with the warp matrix multiply and accumulate instructions.
//!
//! `computeGemm` computes D = alpha * A * B + beta * C. The dimensions of both C and D are
//! `m_global` x `n_global`. A is `m_global` x `k_global` (row-major) and B is `k_global` x
//! `n_global` (column-major). Each work group computes one 128 x 128 tile of the result per
//! iteration. When the tile is computed, the work group stores it to global memory and begins a
//! new iteration, selecting a new 128 x 128 tile to compute. Each work group consists of eight
//! warps. For the 128 x 128 tile, each warp computes eight 16 x 16 subtiles, organized in a 2 x 4
//! two-dimensional array. Warps compute the 16 x 16 subtiles with `wmma.mma` while moving through
//! the K dimension of A and B and accumulating the intermediate result in fragments.
//!
//! There are a number of simple optimizations used in the algorithm:
//! - The work group copies the 128 x 128 tile of C from global memory to shared memory. After
//!   that, each warp loads the fragments of C from shared memory, which avoids random global
//!   memory accesses.
//! - On each internal iteration, the work group copies a portion of A and B to shared memory.
//!   After that, all warps of the work group reuse the data in shared memory, which reduces the
//!   number of copies from global memory.
//! - The portions of A and B are stored in shared memory with an additional padding (skew) to
//!   reduce shared memory bank conflicts.
//! - When the work group finishes computing the tiles of the result, each warp stores its
//!   subtiles to shared memory. The work group then copies the shared memory contents to global
//!   memory, which avoids random global memory accesses.
//! - The work group tile size is chosen to maximize the register utilization, but carefully
//!   enough to avoid local memory use.

const std = @import("std");
const nvptx = std.nvptx;
const wmma = @import("wmma");
const constants = @import("cudaTensorCoreGemm_constants.zig");

const m = constants.m;
const n = constants.n;
const k = constants.k;
const shape: nvptx.matrix.Shape = .{ .m = m, .n = n, .k = k };

const warp_size = constants.warp_size;
const shmem_stride = constants.shmem_stride;
const shmem_offset = constants.shmem_offset;
const global_mem_stride = constants.global_mem_stride;
const k_global = constants.k_global;
const warp_row_tiles = constants.warp_row_tiles;
const warp_col_tiles = constants.warp_col_tiles;
const chunk_k = constants.chunk_k;
const warps_per_block = constants.warps_per_block;
const block_row_tiles = constants.block_row_tiles;
const block_col_tiles = constants.block_col_tiles;
const m_tiles = constants.m_tiles;
const n_tiles = constants.n_tiles;
const k_tiles = constants.k_tiles;
const chunk_copy_line_lanes = constants.chunk_copy_line_lanes;
const chunk_copy_lines_per_warp = constants.chunk_copy_lines_per_warp;
const c_layout: nvptx.matrix.Layout = .row;

const Copy = @Vector(4, u32);

const MatrixA = wmma.Fragment(shape, .{ .a = .f16 });
const MatrixB = wmma.Fragment(shape, .{ .b = .f16 });
const Accumulator = wmma.Fragment(shape, .{ .c = .f32 });

extern var shmem: [0]f16 align(16) addrspace(.shared);

inline fn loadA(pointer: anytype, stride: u32) MatrixA {
    return wmma.loadSyncAligned(shape, .{ .a = .f16 }, .row, pointer, stride);
}

inline fn loadB(pointer: anytype, stride: u32) MatrixB {
    return wmma.loadSyncAligned(shape, .{ .b = .f16 }, .col, pointer, stride);
}

inline fn loadC(pointer: anytype, stride: u32) Accumulator {
    return wmma.loadSyncAligned(shape, .{ .c = .f32 }, c_layout, pointer, stride);
}

inline fn store(pointer: anytype, accumulator: Accumulator, stride: u32) void {
    wmma.storeSyncAligned(shape, .f32, c_layout, pointer, accumulator, stride);
}

inline fn multiply(a: MatrixA, b: MatrixB, c: Accumulator) Accumulator {
    return wmma.mmaSyncAligned(shape, .{ .element = .f16, .accumulator = .f32 }, a, b, c);
}

const SharedCopies = [*]addrspace(.shared) Copy;
const GlobalCopies = [*]addrspace(.global) Copy;
const ConstGlobalCopies = [*]addrspace(.global) const Copy;

export fn computeGemm(
    a: [*]addrspace(.global) const f16,
    b: [*]addrspace(.global) const f16,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta: f32,
) callconv(.kernel) void {
    const shmem_half: [*]addrspace(.shared) f16 = &shmem;
    const shmem_float: [*]addrspace(.shared) f32 = @ptrCast(@alignCast(&shmem));

    const warp_id = @workItemId(0) / warp_size;
    const lane_id = @workItemId(0) % warp_size;

    const shmem_idx_b_off = block_col_tiles * m;

    const shmem_warp_tile_ptr = shmem_float + (warp_id / 2) * shmem_stride * k * 2 +
        (warp_id % 2) * shmem_offset;

    const shmem_warp_stream_ptr = shmem_float + warp_id * shmem_stride * k;

    // The kernel multiplies beta by alpha again at the end of each tile computation. Dividing
    // here may lose precision in general. Zero needs special handling.
    const scaled_beta = beta / alpha;

    var block_pos = @workGroupId(0);
    while (true) : (block_pos += nvptx.nctaid(0)) {
        const block_tile_i = ((block_pos * block_row_tiles) / n_tiles) *
            block_col_tiles;
        const block_tile_j = (block_pos * block_row_tiles) % n_tiles;

        if (block_tile_i >= m_tiles) break;

        const gmem_idx = (block_tile_i + warp_id) * m * global_mem_stride + block_tile_j * n;
        const src_gmem_warp_stream_ptr = c + gmem_idx;

        inline for (0..k) |i| {
            const destination: SharedCopies =
                @ptrCast(@alignCast(shmem_warp_stream_ptr + shmem_stride * i));
            const source: ConstGlobalCopies =
                @ptrCast(@alignCast(src_gmem_warp_stream_ptr + global_mem_stride * i));
            destination[lane_id] = source[lane_id];
        }

        nvptx.barrierSyncAligned(.work_group, null);

        var accumulators: [warp_col_tiles][warp_row_tiles]Accumulator = undefined;

        inline for (0..warp_col_tiles) |i| {
            inline for (0..warp_row_tiles) |j| {
                const tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * k + j * n;
                accumulators[i][j] = loadC(tile_ptr, shmem_stride);
            }
        }

        nvptx.barrierSyncAligned(.work_group, null);

        inline for (0..warp_col_tiles) |i| {
            inline for (0..warp_row_tiles) |j| {
                accumulators[i][j] *= @splat(scaled_beta);
            }
        }

        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const warp_ptr = if (warp_id < 4)
            a + block_tile_i * m * k_global + m * k_global * (warp_id % 4) * 2
        else
            b + block_tile_j * n * k_global + n * k_global * (warp_id % 4) * 2;

        var tile_k: u32 = 0;
        while (tile_k < k_tiles) : (tile_k += chunk_k) {
            // The first half of the warps in the work group copy the A matrix, the rest copy the
            // B matrix.
            var shmem_idx = if (warp_id < warps_per_block / 2)
                m * (warp_id % (warps_per_block / 2)) * 2
            else
                n * (warp_id % (warps_per_block / 2)) * 2 + shmem_idx_b_off;

            // The first half of the warp copies the first row or column of the matrix, the second
            // half of the warp copies the next.
            const lane_line = lane_id / chunk_copy_line_lanes;
            const lane_column = lane_id % chunk_copy_line_lanes;
            var lane_ptr = warp_ptr + tile_k * k + lane_line * k_global +
                lane_column * (@sizeOf(Copy) / @sizeOf(f16));

            // The second half of the warp shifts to the next row or column in shared memory.
            shmem_idx += lane_line;

            inline for (0..((warp_size / 2) / chunk_copy_lines_per_warp) * 2) |_| {
                const destination: SharedCopies =
                    @ptrCast(@alignCast(shmem_half + shmem_idx * row_stride));
                const source: ConstGlobalCopies = @ptrCast(@alignCast(lane_ptr));
                destination[lane_column] = source[0];

                lane_ptr += k_global * chunk_copy_lines_per_warp;
                shmem_idx += chunk_copy_lines_per_warp;
            }

            nvptx.barrierSyncAligned(.work_group, null);

            inline for (0..chunk_k) |k_step| {
                var a_fragments: [warp_col_tiles]MatrixA = undefined;
                var b_fragments: [warp_row_tiles]MatrixB = undefined;

                inline for (0..warp_col_tiles) |i| {
                    const shmem_idx_a = (warp_id / 2) * m * 2 + (i * m);
                    const tile_ptr = shmem_half + shmem_idx_a * row_stride + k_step * k;

                    a_fragments[i] = loadA(tile_ptr, row_stride);

                    inline for (0..warp_row_tiles) |j| {
                        if (i == 0) {
                            // The B matrix fragment is loaded once, because the other A matrix
                            // fragments reuse it.
                            const shmem_idx_b =
                                shmem_idx_b_off + (warp_row_tiles * n) * (warp_id % 2) + (j * n);
                            const b_tile_ptr = shmem_half + shmem_idx_b * row_stride + k_step * k;

                            b_fragments[j] = loadB(b_tile_ptr, row_stride);
                        }

                        accumulators[i][j] =
                            multiply(a_fragments[i], b_fragments[j], accumulators[i][j]);
                    }
                }
            }

            nvptx.barrierSyncAligned(.work_group, null);
        }

        inline for (0..warp_col_tiles) |i| {
            inline for (0..warp_row_tiles) |j| {
                accumulators[i][j] *= @splat(alpha);

                const tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * k + j * n;
                store(tile_ptr, accumulators[i][j], shmem_stride);
            }
        }

        nvptx.barrierSyncAligned(.work_group, null);

        const dst_gmem_warp_stream_ptr = d + gmem_idx;

        inline for (0..k) |i| {
            const destination: GlobalCopies =
                @ptrCast(@alignCast(dst_gmem_warp_stream_ptr + global_mem_stride * i));
            const source: SharedCopies =
                @ptrCast(@alignCast(shmem_warp_stream_ptr + shmem_stride * i));
            destination[lane_id] = source[lane_id];
        }

        nvptx.barrierSyncAligned(.work_group, null);
    }
}

const row_stride = k * chunk_k + constants.skew_half;

/// Performs an M x N x K GEMM (C = alpha * A * B + beta * C) assuming that the matrices are packed
/// in memory, that M, N and K are multiples of 16 and that neither A nor B is transposed. This
/// version does not use shared memory.
export fn simpleWmmaGemm(
    a: [*]addrspace(.global) const f16,
    b: [*]addrspace(.global) const f16,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    m_ld: u32,
    n_ld: u32,
    k_ld: u32,
    alpha: f32,
    beta: f32,
) callconv(.kernel) void {
    const lda = k_ld;
    const ldb = k_ld;
    const ldc = n_ld;

    const warp_m = (@workGroupId(0) * @workGroupSize(0) + @workItemId(0)) / warp_size;
    const warp_n = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);

    var acc_frag: Accumulator = @splat(0);

    var i: u32 = 0;
    while (i < k_ld) : (i += k) {
        const a_col = i;
        const a_row = warp_m * m;
        const b_col = warp_n * n;
        const b_row = i;

        if (a_row < m_ld and a_col < k_ld and b_row < k_ld and b_col < n_ld) {
            const a_frag = loadA(a + a_col + a_row * lda, lda);
            const b_frag = loadB(b + b_row + b_col * ldb, ldb);

            acc_frag = multiply(a_frag, b_frag, acc_frag);
        }
    }

    const c_col = warp_n * n;
    const c_row = warp_m * m;

    if (c_row < m_ld and c_col < n_ld) {
        var c_frag = loadC(c + c_col + c_row * ldc, ldc);

        c_frag =
            @as(Accumulator, @splat(alpha)) * acc_frag + @as(Accumulator, @splat(beta)) * c_frag;

        store(d + c_col + c_row * ldc, c_frag, ldc);
    }
}
