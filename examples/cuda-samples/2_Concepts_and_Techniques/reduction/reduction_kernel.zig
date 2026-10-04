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

//! Parallel reduction kernels.

const std = @import("std");
const common = @import("reduction_common.zig");

extern var shared_memory: [0]u8 align(16) addrspace(.shared);

inline fn sharedMemory(comptime T: type) [*]addrspace(.shared) T {
    return @ptrCast(&shared_memory);
}

inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (@typeInfo(@TypeOf(a)) == .int) a +% b else a + b;
}

inline fn shflDown(comptime T: type, value: T, offset: u32, member_mask: std.nvptx.MemberMask) T {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.down, std.nvptx.warp_size);
    if (@bitSizeOf(T) == 32) return std.nvptx.shflSync(.down, T, value, offset, clamp, member_mask);
    const words: [2]u32 = @bitCast(value);
    return @bitCast([2]u32{
        std.nvptx.shflSync(.down, u32, words[0], offset, clamp, member_mask),
        std.nvptx.shflSync(.down, u32, words[1], offset, clamp, member_mask),
    });
}

inline fn shflXor(comptime T: type, value: T, lane_mask: u32) T {
    const clamp: std.nvptx.ShuffleClamp = .fromWidth(.bfly, std.nvptx.warp_size);
    if (@bitSizeOf(T) == 32) {
        return std.nvptx.shflSync(.bfly, T, value, lane_mask, clamp, .all);
    }
    const words: [2]u32 = @bitCast(value);
    return @bitCast([2]u32{
        std.nvptx.shflSync(.bfly, u32, words[0], lane_mask, clamp, .all),
        std.nvptx.shflSync(.bfly, u32, words[1], lane_mask, clamp, .all),
    });
}

inline fn warpReduceSum(comptime T: type, member_mask: std.nvptx.MemberMask, my_sum_in: T) T {
    if (T == i32 and std.nvptx.gpu.sm >= 80) {
        return std.nvptx.reduxSync(.add, i32, my_sum_in, member_mask);
    }
    var my_sum = my_sum_in;
    var offset: u32 = std.nvptx.warp_size / 2;
    while (offset > 0) : (offset /= 2) {
        my_sum = add(my_sum, shflDown(T, my_sum, offset, member_mask));
    }
    return my_sum;
}

/// Sums the values of the 32 work items of a warp and returns the sum to every one of them.
inline fn warpReduceAll(comptime T: type, value: T) T {
    if (T == i32 and std.nvptx.gpu.sm >= 80) {
        return std.nvptx.reduxSync(.add, i32, value, .all);
    }
    var sum = value;
    var lane_mask: u32 = std.nvptx.warp_size / 2;
    while (lane_mask > 0) : (lane_mask /= 2) sum = add(sum, shflXor(T, sum, lane_mask));
    return sum;
}

/// Adds `sdata[tid + 32]` when `block_size` is at least 64, then sums the values of the warp with
/// shuffles. The first work item of the warp returns the sum.
inline fn finalWarpSum(
    comptime T: type,
    comptime block_size: u32,
    sdata: [*]addrspace(.shared) T,
    tid: u32,
    my_sum_in: T,
) T {
    var my_sum = my_sum_in;
    if (block_size >= 64) my_sum = add(my_sum, sdata[tid + 32]);
    var offset: u32 = std.nvptx.warp_size / 2;
    while (offset > 0) : (offset /= 2) {
        my_sum = add(my_sum, shflDown(T, my_sum, offset, .all));
    }
    return my_sum;
}

/// Sums with an interleaved addressing that leaves no whole warp active.
fn reduce0(comptime T: type) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

            sdata[tid] = if (i < n) g_idata[i] else 0;

            std.nvptx.barrierSyncAligned(.work_group, null);

            var s: u32 = 1;
            while (s < @workGroupSize(0)) : (s *= 2) {
                if (tid % (2 * s) == 0) {
                    sdata[tid] = add(sdata[tid], sdata[tid + s]);
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid == 0) g_odata[@workGroupId(0)] = sdata[0];
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(0, T, null, null) });
        }
    };
}

/// Sums with contiguous work items, whose interleaved addressing causes shared memory bank
/// conflicts.
fn reduce1(comptime T: type) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

            sdata[tid] = if (i < n) g_idata[i] else 0;

            std.nvptx.barrierSyncAligned(.work_group, null);

            var s: u32 = 1;
            while (s < @workGroupSize(0)) : (s *= 2) {
                const index = 2 * s * tid;

                if (index < @workGroupSize(0)) {
                    sdata[index] = add(sdata[index], sdata[index + s]);
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid == 0) g_odata[@workGroupId(0)] = sdata[0];
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(1, T, null, null) });
        }
    };
}

/// Sums with sequential addressing, which has no divergence and no bank conflicts.
fn reduce2(comptime T: type) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

            sdata[tid] = if (i < n) g_idata[i] else 0;

            std.nvptx.barrierSyncAligned(.work_group, null);

            var s: u32 = @workGroupSize(0) / 2;
            while (s > 0) : (s >>= 1) {
                if (tid < s) {
                    sdata[tid] = add(sdata[tid], sdata[tid + s]);
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid == 0) g_odata[@workGroupId(0)] = sdata[0];
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(2, T, null, null) });
        }
    };
}

/// Uses n/2 work items: the first level of the reduction happens while reading from global
/// memory.
fn reduce3(comptime T: type) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const i = @workGroupId(0) * (@workGroupSize(0) * 2) + @workItemId(0);

            var my_sum: T = if (i < n) g_idata[i] else 0;

            if (i + @workGroupSize(0) < n) my_sum = add(my_sum, g_idata[i + @workGroupSize(0)]);

            sdata[tid] = my_sum;
            std.nvptx.barrierSyncAligned(.work_group, null);

            var s: u32 = @workGroupSize(0) / 2;
            while (s > 0) : (s >>= 1) {
                if (tid < s) {
                    my_sum = add(my_sum, sdata[tid + s]);
                    sdata[tid] = my_sum;
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid == 0) g_odata[@workGroupId(0)] = my_sum;
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(3, T, null, null) });
        }
    };
}

/// Uses warp shuffles for the last 32 values. Needs `64 * @sizeOf(T)` bytes of shared memory when
/// `block_size` is at most 32, and `block_size * @sizeOf(T)` bytes otherwise.
fn reduce4(comptime T: type, comptime block_size: u32) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const i = @workGroupId(0) * (@workGroupSize(0) * 2) + @workItemId(0);

            var my_sum: T = if (i < n) g_idata[i] else 0;

            if (i + block_size < n) my_sum = add(my_sum, g_idata[i + block_size]);

            sdata[tid] = my_sum;
            std.nvptx.barrierSyncAligned(.work_group, null);

            var s: u32 = @workGroupSize(0) / 2;
            while (s > 32) : (s >>= 1) {
                if (tid < s) {
                    my_sum = add(my_sum, sdata[tid + s]);
                    sdata[tid] = my_sum;
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid < 32) my_sum = finalWarpSum(T, block_size, sdata, tid, my_sum);

            if (tid == 0) g_odata[@workGroupId(0)] = my_sum;
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(4, T, block_size, null) });
        }
    };
}

/// Unrolls the shared memory steps for `block_size` work items per work group.
fn reduce5(comptime T: type, comptime block_size: u32) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const i = @workGroupId(0) * (block_size * 2) + @workItemId(0);

            var my_sum: T = if (i < n) g_idata[i] else 0;

            if (i + block_size < n) my_sum = add(my_sum, g_idata[i + block_size]);

            sdata[tid] = my_sum;
            std.nvptx.barrierSyncAligned(.work_group, null);

            inline for (.{ 256, 128, 64 }) |step| {
                if (block_size >= 2 * step and tid < step) {
                    my_sum = add(my_sum, sdata[tid + step]);
                    sdata[tid] = my_sum;
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid < 32) my_sum = finalWarpSum(T, block_size, sdata, tid, my_sum);

            if (tid == 0) g_odata[@workGroupId(0)] = my_sum;
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(5, T, block_size, null) });
        }
    };
}

/// Adds several elements per work item sequentially. The number of elements depends on the
/// number of work groups.
fn reduce6(comptime T: type, comptime block_size: u32, comptime n_is_pow2: bool) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            var grid_size: u32 = block_size * std.nvptx.nctaid(0);

            var my_sum: T = 0;

            if (n_is_pow2) {
                var i = @workGroupId(0) * block_size * 2 + @workItemId(0);
                grid_size = grid_size << 1;

                while (i < n) {
                    my_sum = add(my_sum, g_idata[i]);
                    if (i + block_size < n) my_sum = add(my_sum, g_idata[i + block_size]);
                    i += grid_size;
                }
            } else {
                var i = @workGroupId(0) * block_size + @workItemId(0);
                while (i < n) {
                    my_sum = add(my_sum, g_idata[i]);
                    i += grid_size;
                }
            }

            sdata[tid] = my_sum;
            std.nvptx.barrierSyncAligned(.work_group, null);

            inline for (.{ 256, 128, 64 }) |step| {
                if (block_size >= 2 * step and tid < step) {
                    my_sum = add(my_sum, sdata[tid + step]);
                    sdata[tid] = my_sum;
                }

                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            if (tid < 32) my_sum = finalWarpSum(T, block_size, sdata, tid, my_sum);

            if (tid == 0) g_odata[@workGroupId(0)] = my_sum;
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(6, T, block_size, n_is_pow2) });
        }
    };
}

/// Reduces within each warp first. Needs `block_size / 32 + 1` elements of shared memory.
fn reduce7(comptime T: type, comptime block_size: u32, comptime n_is_pow2: bool) type {
    return struct {
        fn kernel(
            noalias g_idata: [*]addrspace(.global) const T,
            noalias g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            var grid_size: u32 = block_size * std.nvptx.nctaid(0);
            const mask_length = if (block_size & 31 > 0) 32 - (block_size & 31) else 0;
            const mask: std.nvptx.MemberMask = @fromBackingInt(0xffffffff >> mask_length);

            var my_sum: T = 0;

            if (n_is_pow2) {
                var i = @workGroupId(0) * block_size * 2 + @workItemId(0);
                grid_size = grid_size << 1;

                while (i < n) {
                    my_sum = add(my_sum, g_idata[i]);
                    if (i + block_size < n) my_sum = add(my_sum, g_idata[i + block_size]);
                    i += grid_size;
                }
            } else {
                var i = @workGroupId(0) * block_size + @workItemId(0);
                while (i < n) {
                    my_sum = add(my_sum, g_idata[i]);
                    i += grid_size;
                }
            }

            my_sum = warpReduceSum(T, mask, my_sum);

            if (tid % std.nvptx.warp_size == 0) sdata[tid / std.nvptx.warp_size] = my_sum;

            std.nvptx.barrierSyncAligned(.work_group, null);

            const shmem_extent = @max(block_size / std.nvptx.warp_size, 1);
            const ballot_result = std.nvptx.voteBallotSync(tid < shmem_extent, mask);
            if (tid < shmem_extent) {
                my_sum = sdata[tid];
                my_sum = warpReduceSum(T, @fromBackingInt(ballot_result), my_sum);
            }

            if (tid == 0) g_odata[@workGroupId(0)] = my_sum;
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(7, T, block_size, n_is_pow2) });
        }
    };
}

/// Reduces with warp-wide sums instead of shared memory steps.
fn cgReduce(comptime T: type) type {
    return struct {
        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const cta_size = @workGroupSize(0);
            const num_ctas = std.nvptx.nctaid(0);
            const thread_rank = @workItemId(0);
            const thread_index = @workGroupId(0) * cta_size + thread_rank;

            var thread_val: T = 0;
            {
                var i = thread_index;
                const index_stride = num_ctas * cta_size;
                while (i < n) {
                    thread_val = add(thread_val, g_idata[i]);
                    i += index_stride;
                }
                sdata[thread_rank] = thread_val;
            }

            {
                var cta_index = cta_size >> 1;
                while (cta_index >= 32) {
                    std.nvptx.barrierSyncAligned(.work_group, null);
                    if (thread_rank < cta_index) {
                        thread_val = add(thread_val, sdata[thread_rank + cta_index]);
                        sdata[thread_rank] = thread_val;
                    }
                    cta_index >>= 1;
                }
            }

            {
                std.nvptx.barrierSyncAligned(.work_group, null);
                if (thread_rank / std.nvptx.warp_size == 0) thread_val = warpReduceAll(T, thread_val);
            }

            if (thread_rank == 0) g_odata[@workGroupId(0)] = thread_val;
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(8, T, null, null) });
        }
    };
}

/// Reduces with groups of `multi_warp_group_size` work items that span several warps.
fn multiWarpCgReduce(
    comptime T: type,
    comptime block_size: u32,
    comptime multi_warp_group_size: u32,
) type {
    return struct {
        var warp_sums: [block_size / std.nvptx.warp_size]T addrspace(.shared) = undefined;

        fn kernel(
            g_idata: [*]addrspace(.global) const T,
            g_odata: [*]addrspace(.global) T,
            n: u32,
        ) callconv(.kernel) void {
            const sdata = sharedMemory(T);

            const tid = @workItemId(0);
            const warps_per_group = multi_warp_group_size / std.nvptx.warp_size;
            var grid_size: u32 = block_size * std.nvptx.nctaid(0);
            var thread_val: T = 0;

            if (n & (n -% 1) == 0) {
                var i = @workGroupId(0) * block_size * 2 + @workItemId(0);
                grid_size = grid_size << 1;

                while (i < n) {
                    thread_val = add(thread_val, g_idata[i]);
                    if (i + block_size < n) {
                        thread_val = add(thread_val, g_idata[i + @workGroupSize(0)]);
                    }
                    i += grid_size;
                }
            } else {
                var i = @workGroupId(0) * block_size + @workItemId(0);
                while (i < n) {
                    thread_val = add(thread_val, g_idata[i]);
                    i += grid_size;
                }
            }

            const warp = tid / std.nvptx.warp_size;
            const lane = tid % std.nvptx.warp_size;
            const warp_sum = warpReduceAll(T, thread_val);
            if (lane == 0) warp_sums[warp] = warp_sum;
            std.nvptx.barrierSyncAligned(.work_group, null);
            const group_first_warp = tid / multi_warp_group_size * warps_per_group;
            thread_val = 0;
            inline for (0..warps_per_group) |k| {
                thread_val = add(thread_val, warp_sums[group_first_warp + k]);
            }

            if (tid % multi_warp_group_size == 0) sdata[tid / multi_warp_group_size] = thread_val;
            std.nvptx.barrierSyncAligned(.work_group, null);

            if (tid == 0) {
                thread_val = 0;
                for (0..block_size / multi_warp_group_size) |i| {
                    thread_val = add(thread_val, sdata[i]);
                }
                g_odata[@workGroupId(0)] = thread_val;
            }
        }

        comptime {
            @export(&kernel, .{ .name = common.kernelName(9, T, block_size, null) });
        }
    };
}

comptime {
    for ([_]type{ i32, f32, f64 }) |T| {
        _ = reduce0(T);
        _ = reduce1(T);
        _ = reduce2(T);
        _ = reduce3(T);
        for (common.block_sizes) |block_size| {
            _ = reduce4(T, block_size);
            _ = reduce5(T, block_size);
            _ = reduce6(T, block_size, true);
            _ = reduce6(T, block_size, false);
        }
        for (common.block_sizes_7) |block_size| {
            _ = reduce7(T, block_size, true);
            _ = reduce7(T, block_size, false);
        }
        _ = cgReduce(T);
        for (common.multi_warp_block_sizes) |block_size| {
            _ = multiWarpCgReduce(T, block_size, block_size / common.multi_warp_groups);
        }
    }
}
