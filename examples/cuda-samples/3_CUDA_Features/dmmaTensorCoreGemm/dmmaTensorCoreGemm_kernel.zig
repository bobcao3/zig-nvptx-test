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

//! The kernels of a double precision GEMM computation with the warp matrix multiply and
//! accumulate instructions.
//!
//! The kernels compute D = alpha * A * B + beta * C. The dimensions of both C and D are
//! `m_global` x `n_global`. A is `m_global` x `k_global` (row-major) and B is `k_global` x
//! `n_global` (column-major). Each work group computes one 64 x 64 tile of the result per
//! iteration. When the tile is computed, the work group stores it to global memory and begins a
//! new iteration, selecting a new 64 x 64 tile to compute. Each work group consists of eight
//! warps. For the 64 x 64 tile, each warp computes eight 8 x 8 subtiles, organized in a 2 x 4
//! two-dimensional array. Warps compute the 8 x 8 subtiles with `wmma.mma` while moving through
//! the K dimension of A and B and accumulating the intermediate result in fragments.
//!
//! There are a number of simple optimizations used in the algorithm:
//! - The work group copies the 64 x 64 tile of C from global memory to shared memory. After
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
const cp_async = nvptx.cp_async;
const constants = @import("dmmaTensorCoreGemm_constants.zig");

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
const block_row_warps = constants.block_row_warps;
const m_tiles = constants.m_tiles;
const n_tiles = constants.n_tiles;
const k_tiles = constants.k_tiles;
const chunk_copy_line_lanes = constants.chunk_copy_line_lanes;
const chunk_copy_lines_per_warp = constants.chunk_copy_lines_per_warp;
const c_layout: nvptx.matrix.Layout = .row;

const Copy = @Vector(2, f64);
const copy_elements = @sizeOf(Copy) / @sizeOf(f64);
const SharedDoubles = [*]addrspace(.shared) f64;

const MatrixA = wmma.Fragment(shape, .{ .a = .f64 });
const MatrixB = wmma.Fragment(shape, .{ .b = .f64 });
const Accumulator = wmma.Fragment(shape, .{ .c = .f64 });
const Accumulators = [warp_col_tiles][warp_row_tiles]Accumulator;

extern var shmem: [0]f64 align(16) addrspace(.shared);

inline fn loadA(pointer: anytype, stride: u32) MatrixA {
    return wmma.loadSyncAligned(shape, .{ .a = .f64 }, .row, pointer, stride);
}

inline fn loadB(pointer: anytype, stride: u32) MatrixB {
    return wmma.loadSyncAligned(shape, .{ .b = .f64 }, .col, pointer, stride);
}

inline fn loadC(pointer: anytype, stride: u32) Accumulator {
    return wmma.loadSyncAligned(shape, .{ .c = .f64 }, c_layout, pointer, stride);
}

inline fn store(pointer: anytype, accumulator: Accumulator, stride: u32) void {
    wmma.storeSyncAligned(shape, .f64, c_layout, pointer, accumulator, stride);
}

inline fn multiply(a: MatrixA, b: MatrixB, c: Accumulator) Accumulator {
    return wmma.mmaSyncAligned(shape, .{ .element = .f64, .accumulator = .f64 }, a, b, c);
}

const row_stride = k * chunk_k + constants.skew_double;

const shmem_idx_b_off = block_col_tiles * m;

inline fn sharedCopy(pointer: SharedDoubles) *addrspace(.shared) Copy {
    return @ptrCast(@alignCast(pointer));
}

inline fn globalCopy(pointer: [*]addrspace(.global) const f64) *addrspace(.global) const Copy {
    return @ptrCast(@alignCast(pointer));
}

inline fn loadAccumulators(accumulators: *Accumulators, shmem_warp_tile_ptr: SharedDoubles) void {
    inline for (0..warp_col_tiles) |i| {
        inline for (0..warp_row_tiles) |j| {
            const tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * n + j * n;
            accumulators[i][j] = loadC(tile_ptr, shmem_stride);
        }
    }
}

inline fn scaleAccumulators(accumulators: *Accumulators, factor: f64) void {
    inline for (0..warp_col_tiles) |i| {
        inline for (0..warp_row_tiles) |j| {
            accumulators[i][j] *= @splat(factor);
        }
    }
}

/// Multiplies the fragments of the A and B tiles that sit in shared memory and adds the products
/// to `accumulators`.
inline fn multiplyChunk(
    accumulators: *Accumulators,
    shmem_double: SharedDoubles,
    warp_id: u32,
) void {
    inline for (0..chunk_k) |k_step| {
        var a_fragments: [warp_col_tiles]MatrixA = undefined;
        var b_fragments: [warp_row_tiles]MatrixB = undefined;

        inline for (0..warp_col_tiles) |i| {
            const shmem_idx_a = (warp_id / 2) * m * 2 + (i * m);
            const tile_ptr = shmem_double + shmem_idx_a * row_stride + k_step * k;

            a_fragments[i] = loadA(tile_ptr, row_stride);

            inline for (0..warp_row_tiles) |j| {
                if (i == 0) {
                    // The B matrix fragment is loaded once, because the other A matrix fragments
                    // reuse it.
                    const shmem_idx_b =
                        shmem_idx_b_off + (warp_row_tiles * n) * (warp_id % 2) + (j * n);
                    const b_tile_ptr = shmem_double + shmem_idx_b * row_stride + k_step * k;

                    b_fragments[j] = loadB(b_tile_ptr, row_stride);
                }

                accumulators[i][j] = multiply(a_fragments[i], b_fragments[j], accumulators[i][j]);
            }
        }
    }
}

inline fn storeAccumulators(
    accumulators: *Accumulators,
    shmem_warp_tile_ptr: SharedDoubles,
    alpha: f64,
) void {
    inline for (0..warp_col_tiles) |i| {
        inline for (0..warp_row_tiles) |j| {
            accumulators[i][j] *= @splat(alpha);

            const tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * n + j * n;
            store(tile_ptr, accumulators[i][j], shmem_stride);
        }
    }
}

inline fn streamTilesToGlobal(
    dst_gmem_warp_stream_ptr: [*]addrspace(.global) f64,
    shmem_warp_stream_ptr: SharedDoubles,
    lane_id: u32,
) void {
    inline for (0..n) |i| {
        const destination: [*]addrspace(.global) Copy =
            @ptrCast(@alignCast(dst_gmem_warp_stream_ptr + global_mem_stride * i));
        const source: [*]addrspace(.shared) Copy =
            @ptrCast(@alignCast(shmem_warp_stream_ptr + shmem_stride * i));
        destination[lane_id] = source[lane_id];
    }
}

export fn computeDgemm(
    a: [*]addrspace(.global) const f64,
    b: [*]addrspace(.global) const f64,
    c: [*]addrspace(.global) const f64,
    d: [*]addrspace(.global) f64,
    alpha: f64,
    beta: f64,
) callconv(.kernel) void {
    @setEvalBranchQuota(10_000);
    const shmem_double: SharedDoubles = &shmem;

    const warp_id = @workItemId(0) / warp_size;
    const lane_id = @workItemId(0) % warp_size;

    const shmem_warp_tile_ptr = shmem_double +
        (warp_id / block_row_warps) * shmem_stride * n * block_row_warps +
        (warp_id % block_row_warps) * shmem_offset;

    const shmem_warp_stream_ptr = shmem_double + warp_id * shmem_stride * n;

    // The kernel multiplies beta by alpha again at the end of each tile computation. Dividing
    // here may lose precision in general. Zero needs special handling.
    const scaled_beta = beta / alpha;

    var block_pos = @workGroupId(0);
    while (true) : (block_pos += nvptx.nctaid(0)) {
        const block_tile_i = ((block_pos * block_row_tiles) / n_tiles) *
            block_col_tiles;
        const block_tile_j = (block_pos * block_col_tiles) % n_tiles;

        if (block_tile_i >= m_tiles) break;

        const gmem_idx = (block_tile_i + warp_id) * m * global_mem_stride + block_tile_j * n;
        const src_gmem_warp_stream_ptr = c + gmem_idx;

        inline for (0..n) |i| {
            const destination: [*]addrspace(.shared) Copy =
                @ptrCast(@alignCast(shmem_warp_stream_ptr + shmem_stride * i));
            const source: [*]addrspace(.global) const Copy =
                @ptrCast(@alignCast(src_gmem_warp_stream_ptr + global_mem_stride * i));
            destination[lane_id] = source[lane_id];
        }

        nvptx.barrierSyncAligned(.work_group, null);

        var accumulators: Accumulators = undefined;
        loadAccumulators(&accumulators, shmem_warp_tile_ptr);

        nvptx.barrierSyncAligned(.work_group, null);

        scaleAccumulators(&accumulators, scaled_beta);

        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const warp_ptr = if (warp_id < warps_per_block / 2)
            a + block_tile_i * m * k_global + m * k_global * (warp_id % (warps_per_block / 2)) * 2
        else
            b + block_tile_j * n * k_global + n * k_global * (warp_id % (warps_per_block / 2)) * 2;

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
            var lane_ptr = warp_ptr + tile_k * k + lane_line * k_global;

            // The second half of the warp shifts to the next row or column in shared memory.
            shmem_idx += lane_line;

            inline for (0..(warp_size / 2) / chunk_copy_lines_per_warp) |_| {
                const destination: [*]addrspace(.shared) Copy =
                    @ptrCast(@alignCast(shmem_double + shmem_idx * row_stride));
                const source: [*]addrspace(.global) const Copy = @ptrCast(@alignCast(lane_ptr));
                destination[lane_column] = source[lane_column];

                lane_ptr += k_global * chunk_copy_lines_per_warp;
                shmem_idx += chunk_copy_lines_per_warp;
            }

            nvptx.barrierSyncAligned(.work_group, null);

            multiplyChunk(&accumulators, shmem_double, warp_id);

            nvptx.barrierSyncAligned(.work_group, null);
        }

        storeAccumulators(&accumulators, shmem_warp_tile_ptr, alpha);

        nvptx.barrierSyncAligned(.work_group, null);

        streamTilesToGlobal(d + gmem_idx, shmem_warp_stream_ptr, lane_id);

        nvptx.barrierSyncAligned(.work_group, null);
    }
}

export fn computeDgemmAsyncCopy(
    a: [*]addrspace(.global) const f64,
    b: [*]addrspace(.global) const f64,
    c: [*]addrspace(.global) const f64,
    d: [*]addrspace(.global) f64,
    alpha: f64,
    beta: f64,
) callconv(.kernel) void {
    @setEvalBranchQuota(10_000);
    const shmem_double: SharedDoubles = &shmem;

    const warp_id = @workItemId(0) / warp_size;
    const lane_id = @workItemId(0) % warp_size;

    const shmem_warp_tile_ptr = shmem_double +
        (warp_id / block_row_warps) * shmem_stride * n * block_row_warps +
        (warp_id % block_row_warps) * shmem_offset;

    const shmem_warp_stream_ptr = shmem_double + warp_id * shmem_stride * n;

    const scaled_beta = beta / alpha;

    // Each copy moves 2 doubles, so the work item index shifts left by 1.
    const load_stride = 1;

    var block_pos = @workGroupId(0);
    while (true) : (block_pos += nvptx.nctaid(0)) {
        const block_tile_i = ((block_pos * block_row_tiles) / n_tiles) *
            block_col_tiles;
        const block_tile_j = (block_pos * block_col_tiles) % n_tiles;

        if (block_tile_i >= m_tiles) break;

        const gmem_idx = (block_tile_i + warp_id) * m * global_mem_stride + block_tile_j * n;
        const src_gmem_warp_stream_ptr = c + gmem_idx;

        const lane_offset = lane_id << load_stride;
        inline for (0..n) |i| {
            cp_async.copy(
                .cache_global_level,
                sharedCopy(shmem_warp_stream_ptr + (shmem_stride * i) + lane_offset),
                globalCopy(src_gmem_warp_stream_ptr + (global_mem_stride * i) + lane_offset),
            );
            cp_async.commitGroup();
        }
        // Waits for all the copies issued above.
        cp_async.waitGroup(0);
        nvptx.barrierSyncAligned(.work_group, null);

        var accumulators: Accumulators = undefined;
        inline for (0..warp_col_tiles) |i| {
            inline for (0..warp_row_tiles) |j| {
                const tile_ptr = shmem_warp_tile_ptr + i * shmem_stride * n + j * n;

                accumulators[i][j] = loadC(tile_ptr, shmem_stride);
                accumulators[i][j] *= @splat(scaled_beta);
            }
        }

        // The shared memory holds the A and B matrices from here on.
        nvptx.barrierSyncAligned(.work_group, null);

        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const warp_ptr = if (warp_id < warps_per_block / 2)
            a + block_tile_i * m * k_global + m * k_global * (warp_id % (warps_per_block / 2)) * 2
        else
            b + block_tile_j * n * k_global + n * k_global * (warp_id % (warps_per_block / 2)) * 2;

        const stride_per_lane_copy = lane_id / chunk_copy_line_lanes;
        const chunks_per_lane = (warp_size / 2) / chunk_copy_lines_per_warp;
        const lane_load_elem = (lane_id % chunk_copy_line_lanes) << load_stride;

        var tile_k: u32 = 0;
        while (tile_k < k_tiles) : (tile_k += chunk_k) {
            // The first half of the warps in the work group copy the A matrix, the rest copy the
            // B matrix. Because M equals N, the offset `shmem_idx_b_off` serves warps 4-7.
            var shmem_idx = (m * (warp_id % (warps_per_block / 2)) * 2) +
                (shmem_idx_b_off * (warp_id / (warps_per_block / 2)));

            // The first half of the warp copies the first row or column of the matrix, the second
            // half of the warp copies the next.
            var lane_ptr = warp_ptr + tile_k * k + stride_per_lane_copy * k_global + lane_load_elem;

            // The second half of the warp shifts to the next row or column in shared memory.
            shmem_idx += stride_per_lane_copy;
            inline for (0..chunks_per_lane) |_| {
                cp_async.copy(
                    .cache_global_level,
                    sharedCopy(shmem_double + shmem_idx * row_stride + lane_load_elem),
                    globalCopy(lane_ptr),
                );
                cp_async.commitGroup();

                lane_ptr += k_global * chunk_copy_lines_per_warp;
                shmem_idx += chunk_copy_lines_per_warp;
            }

            cp_async.waitGroup(0);
            nvptx.barrierSyncAligned(.work_group, null);

            multiplyChunk(&accumulators, shmem_double, warp_id);

            nvptx.barrierSyncAligned(.work_group, null);
        }

        storeAccumulators(&accumulators, shmem_warp_tile_ptr, alpha);

        nvptx.barrierSyncAligned(.work_group, null);

        streamTilesToGlobal(d + gmem_idx, shmem_warp_stream_ptr, lane_id);

        nvptx.barrierSyncAligned(.work_group, null);
    }
}

export fn computeDgemmCgAsyncCopy(
    a: [*]addrspace(.global) const f64,
    b: [*]addrspace(.global) const f64,
    c: [*]addrspace(.global) const f64,
    d: [*]addrspace(.global) f64,
    alpha: f64,
    beta: f64,
) callconv(.kernel) void {
    @setEvalBranchQuota(10_000);
    const shmem_double: SharedDoubles = &shmem;

    const warp_id = @workItemId(0) / warp_size;
    const lane_id = @workItemId(0) % warp_size;

    const shmem_warp_tile_ptr = shmem_double +
        (warp_id / 2) * shmem_stride * n * 2 + (warp_id % 2) * shmem_offset;

    const shmem_warp_stream_ptr = shmem_double + warp_id * shmem_stride * n;

    const scaled_beta = beta / alpha;

    var block_pos = @workGroupId(0);
    while (true) : (block_pos += nvptx.nctaid(0)) {
        const block_tile_i = ((block_pos * block_row_tiles) / n_tiles) *
            block_col_tiles;
        const block_tile_j = (block_pos * block_col_tiles) % n_tiles;

        if (block_tile_i >= m_tiles) break;

        const gmem_idx = (block_tile_i + warp_id) * m * global_mem_stride + block_tile_j * n;
        const src_gmem_warp_stream_ptr = c + gmem_idx;

        // The 32 work items of the warp copy one row of the tile, 16 bytes each.
        const lane_offset = lane_id * copy_elements;
        inline for (0..n) |i| {
            cp_async.copy(
                .cache_global_level,
                sharedCopy(shmem_warp_stream_ptr + shmem_stride * i + lane_offset),
                globalCopy(src_gmem_warp_stream_ptr + global_mem_stride * i + lane_offset),
            );
            cp_async.commitGroup();
        }

        cp_async.waitGroup(0);
        nvptx.barrierSyncAligned(.work_group, null);

        var accumulators: Accumulators = undefined;
        loadAccumulators(&accumulators, shmem_warp_tile_ptr);

        scaleAccumulators(&accumulators, scaled_beta);

        // The shared memory holds the A and B matrices from here on.
        cp_async.waitGroup(0);
        nvptx.barrierSyncAligned(.work_group, null);

        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const warp_ptr = if (warp_id < 4)
            a + block_tile_i * m * k_global + m * k_global * (warp_id % (warps_per_block / 2)) * 2
        else
            b + block_tile_j * n * k_global + n * k_global * (warp_id % (warps_per_block / 2)) * 2;

        const stride_per_lane_copy = lane_id / chunk_copy_line_lanes;
        const lane_column = lane_id % chunk_copy_line_lanes;
        const chunks_per_lane = (warp_size / 2) / chunk_copy_lines_per_warp;

        var tile_k: u32 = 0;
        while (tile_k < k_tiles) : (tile_k += chunk_k) {
            var shmem_idx = (m * (warp_id % (warps_per_block / 2)) * 2) +
                (shmem_idx_b_off * (warp_id / (warps_per_block / 2)));

            var lane_ptr = warp_ptr + tile_k * k + stride_per_lane_copy * k_global;

            shmem_idx += stride_per_lane_copy;

            inline for (0..chunks_per_lane) |_| {
                // The work items of a tile of `chunk_copy_line_lanes` copy one line, 16 bytes each.
                cp_async.copy(
                    .cache_global_level,
                    sharedCopy(shmem_double + shmem_idx * row_stride + lane_column * copy_elements),
                    globalCopy(lane_ptr + lane_column * copy_elements),
                );
                cp_async.commitGroup();

                lane_ptr += k_global * chunk_copy_lines_per_warp;
                shmem_idx += chunk_copy_lines_per_warp;
            }
            cp_async.waitGroup(0);
            nvptx.barrierSyncAligned(.work_group, null);

            multiplyChunk(&accumulators, shmem_double, warp_id);

            nvptx.barrierSyncAligned(.work_group, null);
        }

        storeAccumulators(&accumulators, shmem_warp_tile_ptr, alpha);

        nvptx.barrierSyncAligned(.work_group, null);

        streamTilesToGlobal(d + gmem_idx, shmem_warp_stream_ptr, lane_id);

        nvptx.barrierSyncAligned(.work_group, null);
    }
}

/// Performs an M x N x K DGEMM (C = alpha * A * B + beta * C) assuming that the matrices are
/// packed in memory, that M, N and K are multiples of 8, 8 and 4, and that A is row-major and B is
/// column-major. This version does not
/// use shared memory.
export fn simpleWmmaGemm(
    a: [*]addrspace(.global) const f64,
    b: [*]addrspace(.global) const f64,
    c: [*]addrspace(.global) const f64,
    d: [*]addrspace(.global) f64,
    m_ld: u32,
    n_ld: u32,
    k_ld: u32,
    alpha: f64,
    beta: f64,
) callconv(.kernel) void {
    @setEvalBranchQuota(10_000);
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
