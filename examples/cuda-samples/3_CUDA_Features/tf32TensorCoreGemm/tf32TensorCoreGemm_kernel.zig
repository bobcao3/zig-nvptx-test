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

//! The kernels of the tf32 matrix multiplication D = alpha * A * B + beta * C. A is row-major
//! and B is column-major.

const std = @import("std");
const nvptx = std.nvptx;
const wmma = @import("wmma");
const cp_async = nvptx.cp_async;
const config = @import("tf32TensorCoreGemm_config.zig");
const Int4 = config.Int4;
const Float4 = config.Float4;

const shape: nvptx.matrix.Shape = .{ .m = config.m, .n = config.n, .k = config.k };
const AFragment = wmma.Fragment(shape, .{ .a = .tf32 });
const BFragment = wmma.Fragment(shape, .{ .b = .tf32 });
const AccumulatorFragment = wmma.Fragment(shape, .{ .c = .f32 });
const mma_options: wmma.Options = .{ .element = .tf32, .accumulator = .f32 };

const shmem_row_elements = config.chunk_k * config.k + config.skew_float;

inline fn roundToTf32(fragment: anytype) @TypeOf(fragment) {
    var result = fragment;
    inline for (0..@typeInfo(@TypeOf(fragment)).vector.len) |t| {
        result[t] = @backingInt(nvptx.Tf32.fromF32(@bitCast(fragment[t])));
    }
    return result;
}

extern var shmem: [0]f32 align(16) addrspace(.shared);

inline fn shmemFloat() [*]addrspace(.shared) f32 {
    return &shmem;
}

inline fn globalRow(pointer: [*]addrspace(.global) const f32) [*]addrspace(.global) const Int4 {
    return @ptrCast(@alignCast(pointer));
}

export fn computeTf32gemm(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta_in: f32,
) callconv(.kernel) void {
    @setEvalBranchQuota(10_000);
    const shmem_f32 = shmemFloat();

    const warp_id: usize = @workItemId(0) / config.warp_size;
    const lane_id: usize = @workItemId(0) % config.warp_size;

    const shmem_idx_b_off = config.block_col_tiles * config.m;

    const shmem_warp_tile_ptr = shmem_f32 +
        (warp_id / config.block_row_warps) * config.shmem_stride * config.n * config.block_row_warps +
        (warp_id % config.block_row_warps) * config.shmem_offset;

    const shmem_warp_stream_ptr = shmem_f32 + warp_id * config.shmem_stride * config.n;

    const beta = beta_in / alpha;

    var block_pos: usize = @workGroupId(0);
    while (true) : (block_pos += nvptx.nctaid(0)) {
        const block_tile_i = ((block_pos * config.block_row_tiles) / config.n_tiles) * config.block_col_tiles;
        const block_tile_j = (block_pos * config.block_col_tiles) % config.n_tiles;

        if (block_tile_i >= config.m_tiles) break;

        const gmem_idx = (block_tile_i + warp_id) * config.m * config.global_mem_stride +
            block_tile_j * config.n;
        const src_gmem_warp_stream_ptr = c + gmem_idx;

        inline for (0..config.n) |i| {
            const dst: [*]addrspace(.shared) Int4 = @ptrCast(@alignCast(
                shmem_warp_stream_ptr + config.shmem_stride * i,
            ));
            const src: [*]addrspace(.global) const Int4 = @ptrCast(@alignCast(
                src_gmem_warp_stream_ptr + config.global_mem_stride * i,
            ));
            dst[lane_id] = src[lane_id];
        }

        nvptx.barrierSyncAligned(.work_group, null);

        var c_frag: [config.warp_col_tiles][config.warp_row_tiles]AccumulatorFragment = undefined;

        inline for (0..config.warp_col_tiles) |i| {
            inline for (0..config.warp_row_tiles) |j| {
                const tile_ptr = shmem_warp_tile_ptr + i * config.shmem_stride * config.n + j * config.n;
                c_frag[i][j] = wmma.loadSyncAligned(shape, .{ .c = .f32 }, .row, tile_ptr, config.shmem_stride);
            }
        }

        nvptx.barrierSyncAligned(.work_group, null);

        inline for (0..config.warp_col_tiles) |i| {
            inline for (0..config.warp_row_tiles) |j| {
                c_frag[i][j] *= @as(AccumulatorFragment, @splat(beta));
            }
        }

        const warp_ptr = if (warp_id < config.warps_per_block / 2)
            a + block_tile_i * config.m * config.k_global +
                config.m * config.k_global * (warp_id % (config.warps_per_block / 2)) * 2
        else
            b + block_tile_j * config.n * config.k_global +
                config.n * config.k_global * (warp_id % (config.warps_per_block / 2)) * 2;

        var tile_k: usize = 0;
        while (tile_k < config.k_tiles) : (tile_k += config.chunk_k) {
            var shmem_idx: usize = if (warp_id < config.warps_per_block / 2)
                config.m * (warp_id % (config.warps_per_block / 2)) * 2
            else
                config.n * (warp_id % (config.warps_per_block / 2)) * 2 + shmem_idx_b_off;

            var lane_ptr = warp_ptr + tile_k * config.k +
                (lane_id / config.chunk_copy_line_lanes) * config.k_global;

            shmem_idx += lane_id / config.chunk_copy_line_lanes;

            inline for (0..((config.warp_size / 2) / config.chunk_copy_lines_per_warp) * 2) |_| {
                const dst: [*]addrspace(.shared) Int4 = @ptrCast(@alignCast(
                    shmem_f32 + shmem_idx * shmem_row_elements,
                ));
                dst[lane_id % config.chunk_copy_line_lanes] = globalRow(lane_ptr)[lane_id % config.chunk_copy_line_lanes];

                lane_ptr += config.k_global * config.chunk_copy_lines_per_warp;
                shmem_idx += config.chunk_copy_lines_per_warp;
            }

            nvptx.barrierSyncAligned(.work_group, null);

            inline for (0..config.chunk_k) |k_step| {
                var a_frag: [config.warp_col_tiles]AFragment = undefined;
                var b_frag: [config.warp_row_tiles]BFragment = undefined;

                inline for (0..config.warp_col_tiles) |i| {
                    const shmem_idx_a = (warp_id / config.block_row_warps) * config.m * config.block_row_warps +
                        i * config.m;
                    const tile_ptr = shmem_f32 + shmem_idx_a * shmem_row_elements + k_step * config.k;

                    a_frag[i] = roundToTf32(wmma.loadSyncAligned(shape, .{ .a = .tf32 }, .row, tile_ptr, shmem_row_elements));

                    inline for (0..config.warp_row_tiles) |j| {
                        if (i == 0) {
                            const shmem_idx_b = shmem_idx_b_off + (config.warp_row_tiles * config.n) * (warp_id % 2) +
                                j * config.n;
                            const b_tile_ptr = shmem_f32 + shmem_idx_b * shmem_row_elements + k_step * config.k;

                            b_frag[j] = roundToTf32(wmma.loadSyncAligned(shape, .{ .b = .tf32 }, .col, b_tile_ptr, shmem_row_elements));
                        }

                        c_frag[i][j] = wmma.mmaSyncAligned(shape, mma_options, a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }

            nvptx.barrierSyncAligned(.work_group, null);
        }

        inline for (0..config.warp_col_tiles) |i| {
            inline for (0..config.warp_row_tiles) |j| {
                c_frag[i][j] *= @as(AccumulatorFragment, @splat(alpha));

                const tile_ptr = shmem_warp_tile_ptr + i * config.shmem_stride * config.n + j * config.n;
                wmma.storeSyncAligned(shape, .f32, .row, tile_ptr, c_frag[i][j], config.shmem_stride);
            }
        }

        nvptx.barrierSyncAligned(.work_group, null);

        const dst_gmem_warp_stream_ptr = d + gmem_idx;

        inline for (0..config.n) |i| {
            const dst: [*]addrspace(.global) Float4 = @ptrCast(@alignCast(
                dst_gmem_warp_stream_ptr + config.global_mem_stride * i,
            ));
            const src: [*]addrspace(.shared) Float4 = @ptrCast(@alignCast(
                shmem_warp_stream_ptr + config.shmem_stride * i,
            ));
            dst[lane_id] = src[lane_id];
        }

        nvptx.barrierSyncAligned(.work_group, null);
    }
}

export fn computeTf32gemmAsyncCopy(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    alpha: f32,
    beta_in: f32,
) callconv(.kernel) void {
    @setEvalBranchQuota(10_000);
    const shmem_f32 = shmemFloat();

    const warp_id: usize = @workItemId(0) / config.warp_size;
    const lane_id: usize = @workItemId(0) % config.warp_size;

    const shmem_idx_b_off = config.block_col_tiles * config.m;

    const shmem_warp_tile_ptr = shmem_f32 +
        (warp_id / config.block_row_warps) * config.shmem_stride * config.n * config.block_row_warps +
        (warp_id % config.block_row_warps) * config.shmem_offset;

    const shmem_warp_stream_ptr = shmem_f32 + warp_id * config.shmem_stride * config.n;

    const beta = beta_in / alpha;

    const load_stride = 2;

    var block_pos: usize = @workGroupId(0);
    while (true) : (block_pos += nvptx.nctaid(0)) {
        const block_tile_i = ((block_pos * config.block_row_tiles) / config.n_tiles) * config.block_col_tiles;
        const block_tile_j = (block_pos * config.block_col_tiles) % config.n_tiles;

        if (block_tile_i >= config.m_tiles) break;

        const gmem_idx = (block_tile_i + warp_id) * config.m * config.global_mem_stride +
            block_tile_j * config.n;
        const src_gmem_warp_stream_ptr = c + gmem_idx;

        inline for (0..config.n) |i| {
            const dst: *addrspace(.shared) Int4 = @ptrCast(@alignCast(
                shmem_warp_stream_ptr + (config.shmem_stride * i) + (lane_id << load_stride),
            ));
            const src: *addrspace(.global) const Int4 = @ptrCast(@alignCast(
                src_gmem_warp_stream_ptr + (config.global_mem_stride * i) + (lane_id << load_stride),
            ));
            cp_async.copy(.cache_global_level, dst, src);
            cp_async.commitGroup();
        }

        cp_async.waitGroup(0);
        nvptx.barrierSyncAligned(.work_group, null);

        var c_frag: [config.warp_col_tiles][config.warp_row_tiles]AccumulatorFragment = undefined;

        inline for (0..config.warp_col_tiles) |i| {
            inline for (0..config.warp_row_tiles) |j| {
                const tile_ptr = shmem_warp_tile_ptr + i * config.shmem_stride * config.n + j * config.n;
                c_frag[i][j] = wmma.loadSyncAligned(shape, .{ .c = .f32 }, .row, tile_ptr, config.shmem_stride);
                c_frag[i][j] *= @as(AccumulatorFragment, @splat(beta));
            }
        }

        nvptx.barrierSyncAligned(.work_group, null);

        const warp_ptr = if (warp_id < config.warps_per_block / 2)
            a + block_tile_i * config.m * config.k_global +
                config.m * config.k_global * (warp_id % (config.warps_per_block / 2)) * 2
        else
            b + block_tile_j * config.n * config.k_global +
                config.n * config.k_global * (warp_id % (config.warps_per_block / 2)) * 2;

        const chunks_per_lane = ((config.warp_size / 2) / config.chunk_copy_lines_per_warp) * 2;
        const lane_load_elem = (lane_id % config.chunk_copy_line_lanes) << load_stride;
        const stride_per_lane_copy = lane_id / config.chunk_copy_line_lanes;

        var tile_k: usize = 0;
        while (tile_k < config.k_tiles) : (tile_k += config.chunk_k) {
            var shmem_idx: usize = (config.m * (warp_id % (config.warps_per_block / 2)) * 2) +
                ((warp_id / (config.warps_per_block / 2)) * shmem_idx_b_off);

            var lane_ptr = warp_ptr + tile_k * config.k + stride_per_lane_copy * config.k_global + lane_load_elem;

            shmem_idx += stride_per_lane_copy;

            inline for (0..chunks_per_lane) |_| {
                const dst: *addrspace(.shared) Int4 = @ptrCast(@alignCast(
                    shmem_f32 + shmem_idx * shmem_row_elements + lane_load_elem,
                ));
                cp_async.copy(.cache_global_level, dst, @as(*addrspace(.global) const Int4, @ptrCast(@alignCast(lane_ptr))));
                cp_async.commitGroup();
                lane_ptr += config.k_global * config.chunk_copy_lines_per_warp;
                shmem_idx += config.chunk_copy_lines_per_warp;
            }
            cp_async.waitGroup(0);
            nvptx.barrierSyncAligned(.work_group, null);

            inline for (0..config.chunk_k) |k_step| {
                var a_frag: [config.warp_col_tiles]AFragment = undefined;
                var b_frag: [config.warp_row_tiles]BFragment = undefined;

                inline for (0..config.warp_col_tiles) |i| {
                    const shmem_idx_a = (warp_id / config.block_row_warps) * config.m * config.block_row_warps +
                        i * config.m;
                    const tile_ptr = shmem_f32 + shmem_idx_a * shmem_row_elements + k_step * config.k;

                    a_frag[i] = roundToTf32(wmma.loadSyncAligned(shape, .{ .a = .tf32 }, .row, tile_ptr, shmem_row_elements));

                    inline for (0..config.warp_row_tiles) |j| {
                        if (i == 0) {
                            const shmem_idx_b = shmem_idx_b_off + (config.warp_row_tiles * config.n) * (warp_id % 2) +
                                j * config.n;
                            const b_tile_ptr = shmem_f32 + shmem_idx_b * shmem_row_elements + k_step * config.k;

                            b_frag[j] = roundToTf32(wmma.loadSyncAligned(shape, .{ .b = .tf32 }, .col, b_tile_ptr, shmem_row_elements));
                        }

                        c_frag[i][j] = wmma.mmaSyncAligned(shape, mma_options, a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }

            nvptx.barrierSyncAligned(.work_group, null);
        }

        inline for (0..config.warp_col_tiles) |i| {
            inline for (0..config.warp_row_tiles) |j| {
                c_frag[i][j] *= @as(AccumulatorFragment, @splat(alpha));

                const tile_ptr = shmem_warp_tile_ptr + i * config.shmem_stride * config.n + j * config.n;
                wmma.storeSyncAligned(shape, .f32, .row, tile_ptr, c_frag[i][j], config.shmem_stride);
            }
        }

        nvptx.barrierSyncAligned(.work_group, null);

        const dst_gmem_warp_stream_ptr = d + gmem_idx;

        inline for (0..config.n) |i| {
            const dst: [*]addrspace(.global) Int4 = @ptrCast(@alignCast(
                dst_gmem_warp_stream_ptr + config.global_mem_stride * i,
            ));
            const src: [*]addrspace(.shared) Int4 = @ptrCast(@alignCast(
                shmem_warp_stream_ptr + config.shmem_stride * i,
            ));
            dst[lane_id] = src[lane_id];
        }

        nvptx.barrierSyncAligned(.work_group, null);
    }
}

/// Performs an M x N x K tf32 GEMM (C = alpha * A * B + beta * C), assuming that the matrices
/// are packed in memory, that M, N and K are multiples of 16, and that A is row-major and B is
/// column-major. This version does not use shared memory.
export fn simpleWmmaTf32gemm(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) const f32,
    d: [*]addrspace(.global) f32,
    m_ld: i32,
    n_ld: i32,
    k_ld: i32,
    alpha: f32,
    beta: f32,
) callconv(.kernel) void {
    const lda: usize = @intCast(k_ld);
    const ldb: usize = @intCast(k_ld);
    const ldc: usize = @intCast(n_ld);

    const warp_m: i32 = @intCast((@workGroupId(0) * @workGroupSize(0) + @workItemId(0)) / config.warp_size);
    const warp_n: i32 = @intCast(@workGroupId(1) * @workGroupSize(1) + @workItemId(1));

    var acc_frag: AccumulatorFragment = @splat(0);

    var i: i32 = 0;
    while (i < k_ld) : (i += config.k) {
        const a_col = i;
        const a_row = warp_m * config.m;

        const b_col = warp_n * config.n;
        const b_row = i;

        if (a_row < m_ld and a_col < k_ld and b_row < k_ld and b_col < n_ld) {
            const a_frag = roundToTf32(wmma.loadSyncAligned(shape, .{ .a = .tf32 }, .row, a + @as(usize, @intCast(a_col)) + @as(usize, @intCast(a_row)) * lda, @intCast(lda)));
            const b_frag = roundToTf32(wmma.loadSyncAligned(shape, .{ .b = .tf32 }, .col, b + @as(usize, @intCast(b_row)) + @as(usize, @intCast(b_col)) * ldb, @intCast(ldb)));

            acc_frag = wmma.mmaSyncAligned(shape, mma_options, a_frag, b_frag, acc_frag);
        }
    }

    const c_col = warp_n * config.n;
    const c_row = warp_m * config.m;

    if (c_row < m_ld and c_col < n_ld) {
        const offset = @as(usize, @intCast(c_col)) + @as(usize, @intCast(c_row)) * ldc;
        var c_frag = wmma.loadSyncAligned(shape, .{ .c = .f32 }, .row, c + offset, @intCast(ldc));

        c_frag = @as(AccumulatorFragment, @splat(alpha)) * acc_frag + @as(AccumulatorFragment, @splat(beta)) * c_frag;

        wmma.storeSyncAligned(shape, .f32, .row, d + offset, c_frag, @intCast(ldc));
    }
}
