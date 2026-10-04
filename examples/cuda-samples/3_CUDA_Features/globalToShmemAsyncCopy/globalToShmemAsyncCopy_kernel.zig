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

//! The matrix multiplication kernels C = A * B. Kernels with `AsyncCopy` in their name copy tiles
//! from global memory to shared memory asynchronously. `wa` is the width of A and `wb` is the width
//! of B.

const std = @import("std");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const Mbarrier = nvptx.Mbarrier;

const Int4 = @Vector(4, u32);

inline fn sharedInt4(pointer: *addrspace(.shared) f32) *addrspace(.shared) Int4 {
    return @ptrCast(@alignCast(pointer));
}

inline fn globalInt4(pointer: *addrspace(.global) const f32) *addrspace(.global) const Int4 {
    return @ptrCast(@alignCast(pointer));
}

/// A pipeline of one work item. `producerCommit` ends a group of copies, and `consumerWait`
/// waits until the oldest group that no `consumerWait` has waited for is complete.
const Pipeline = struct {
    head: u8 = 0,
    tail: u8 = 0,

    inline fn producerCommit(pipe: *Pipeline) void {
        cp_async.commitGroup();
        pipe.head += 1;
    }

    inline fn consumerWait(pipe: *Pipeline) void {
        if (pipe.head == pipe.tail) return;
        switch (pipe.head - pipe.tail - 1) {
            inline 0...7 => |prior| cp_async.waitGroup(prior),
            else => cp_async.waitGroup(8),
        }
        pipe.tail += 1;
    }
};

inline fn arriveAndWait(barrier: *addrspace(.shared) Mbarrier) void {
    const token = barrier.arrive();
    while (!barrier.testWait(token)) {}
}

/// Multi-stage pipeline with chunks of 16 bytes. Requires `block_size % 4 == 0`.
fn MatrixMulAsyncCopyMultiStageLargeChunk(comptime block_size: u32) type {
    return struct {
        const max_pipeline_stages = 4;

        var as: [max_pipeline_stages][block_size][block_size]f32 align(16) addrspace(.shared) = undefined;
        var bs: [max_pipeline_stages][block_size][block_size]f32 align(16) addrspace(.shared) = undefined;

        fn matrixMulAsyncCopyMultiStageLargeChunk(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            var c_sub: f32 = 0.0;

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            const t4x = tx * 4;
            var pipe: Pipeline = .{};

            var a_index = a_begin;
            var b_index = b_begin;
            var i: u32 = 0;
            var a_stage = a_begin;
            var b_stage = b_begin;
            var i_stage: u32 = 0;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
                i += 1;
            }) {
                while (a_stage <= a_index + a_step * max_pipeline_stages) : ({
                    a_stage += a_step;
                    b_stage += b_step;
                    i_stage += 1;
                }) {
                    if (a_stage <= a_end and t4x < block_size) {
                        const j = i_stage % max_pipeline_stages;
                        cp_async.copy(.cache_global_level, sharedInt4(&as[j][ty][t4x]), globalInt4(&a[a_stage + wa * ty + t4x]));
                        cp_async.copy(.cache_global_level, sharedInt4(&bs[j][ty][t4x]), globalInt4(&b[a_stage + wa * ty + t4x]));
                    }
                    pipe.producerCommit();
                }

                pipe.consumerWait();
                nvptx.barrierSyncAligned(.work_group, null);

                const j = i % max_pipeline_stages;

                inline for (0..block_size) |k| {
                    c_sub += as[j][ty][k] * bs[j][k][tx];
                }
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulAsyncCopyMultiStageLargeChunk, .{ .name = "matrixMulAsyncCopyMultiStageLargeChunk" });
        }
    };
}

/// Single-stage pipeline with chunks of 16 bytes. Requires `block_size % 4 == 0`.
fn MatrixMulAsyncCopyLargeChunk(comptime block_size: u32) type {
    return struct {
        var as: [block_size][block_size]f32 align(16) addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 align(16) addrspace(.shared) = undefined;

        fn matrixMulAsyncCopyLargeChunk(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0.0;

            const t4x = tx * 4;
            var pipe: Pipeline = .{};

            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                if (t4x < block_size) {
                    cp_async.copy(.cache_global_level, sharedInt4(&as[ty][t4x]), globalInt4(&a[a_index + wa * ty + t4x]));
                    cp_async.copy(.cache_global_level, sharedInt4(&bs[ty][t4x]), globalInt4(&b[a_index + wa * ty + t4x]));

                    pipe.producerCommit();
                    pipe.consumerWait();
                }

                nvptx.barrierSyncAligned(.work_group, null);

                inline for (0..block_size) |k| {
                    c_sub += as[ty][k] * bs[k][tx];
                }

                nvptx.barrierSyncAligned(.work_group, null);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulAsyncCopyLargeChunk, .{ .name = "matrixMulAsyncCopyLargeChunk" });
        }
    };
}

/// Single-stage pipeline with chunks of 16 bytes and an arrive-wait barrier. Requires
/// `block_size % 4 == 0`.
fn MatrixMulAsyncCopyLargeChunkAwBarrier(comptime block_size: u32) type {
    return struct {
        var bar: Mbarrier addrspace(.shared) = undefined;
        var as: [block_size][block_size]f32 align(16) addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 align(16) addrspace(.shared) = undefined;

        fn matrixMulAsyncCopyLargeChunkAwBarrier(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            if (tx == 0) {
                Mbarrier.init(&bar, @workGroupSize(0) * @workGroupSize(1));
            }
            nvptx.barrierSyncAligned(.work_group, null);

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0.0;

            const t4x = tx * 4;

            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                if (t4x < block_size) {
                    cp_async.copy(.cache_global_level, sharedInt4(&as[ty][t4x]), globalInt4(&a[a_index + wa * ty + t4x]));
                    cp_async.mbarrierArrive(&bar);
                    cp_async.copy(.cache_global_level, sharedInt4(&bs[ty][t4x]), globalInt4(&b[a_index + wa * ty + t4x]));
                    cp_async.mbarrierArrive(&bar);
                }

                arriveAndWait(&bar);

                inline for (0..block_size) |k| {
                    c_sub += as[ty][k] * bs[k][tx];
                }

                arriveAndWait(&bar);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulAsyncCopyLargeChunkAwBarrier, .{ .name = "matrixMulAsyncCopyLargeChunkAwBarrier" });
        }
    };
}

/// Single-stage pipeline with chunks of 4 bytes.
fn MatrixMulAsyncCopySingleStage(comptime block_size: u32) type {
    return struct {
        var as: [block_size][block_size]f32 addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 addrspace(.shared) = undefined;

        fn matrixMulAsyncCopySingleStage(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0.0;

            var pipe: Pipeline = .{};

            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                {
                    cp_async.copy(.cache_all_levels, &as[ty][tx], &a[a_index + wa * ty + tx]);
                    cp_async.copy(.cache_all_levels, &bs[ty][tx], &b[b_index + wb * ty + tx]);

                    pipe.producerCommit();
                }

                pipe.consumerWait();
                nvptx.barrierSyncAligned(.work_group, null);

                inline for (0..block_size) |k| {
                    c_sub += as[ty][k] * bs[k][tx];
                }

                nvptx.barrierSyncAligned(.work_group, null);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulAsyncCopySingleStage, .{ .name = "matrixMulAsyncCopySingleStage" });
        }
    };
}

/// Multi-stage pipeline with chunks of 4 bytes.
fn MatrixMulAsyncCopyMultiStage(comptime block_size: u32) type {
    return struct {
        const max_pipeline_stages = 4;

        var as: [max_pipeline_stages][block_size][block_size]f32 addrspace(.shared) = undefined;
        var bs: [max_pipeline_stages][block_size][block_size]f32 addrspace(.shared) = undefined;

        fn matrixMulAsyncCopyMultiStage(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            var c_sub: f32 = 0.0;

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var pipe: Pipeline = .{};

            var a_index = a_begin;
            var b_index = b_begin;
            var i: u32 = 0;
            var a_stage = a_begin;
            var b_stage = b_begin;
            var i_stage: u32 = 0;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
                i += 1;
            }) {
                while (a_stage <= a_index + a_step * max_pipeline_stages) : ({
                    a_stage += a_step;
                    b_stage += b_step;
                    i_stage += 1;
                }) {
                    if (a_stage <= a_end) {
                        const j = i_stage % max_pipeline_stages;

                        cp_async.copy(.cache_all_levels, &as[j][ty][tx], &a[a_stage + wa * ty + tx]);
                        cp_async.copy(.cache_all_levels, &bs[j][ty][tx], &b[b_stage + wb * ty + tx]);

                        pipe.producerCommit();
                    }
                }
                pipe.consumerWait();

                nvptx.barrierSyncAligned(.work_group, null);

                const j = i % max_pipeline_stages;

                for (0..block_size) |k| {
                    c_sub += as[j][ty][k] * bs[j][k][tx];
                }
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulAsyncCopyMultiStage, .{ .name = "matrixMulAsyncCopyMultiStage" });
        }
    };
}

/// Multi-stage pipeline with a block-wide shared state. The work items with `ty >= block_size`
/// are the producers: they issue the copies. The other work items are the consumers: they compute
/// the product of the tiles that the producers loaded. The `produced` barrier of a stage expects
/// one arrival from each producer, and the `consumed` barrier expects one from each consumer.
fn MatrixMulAsyncCopyMultiStageSharedState(comptime block_size_x: u32) type {
    return struct {
        const max_pipeline_stages = 4;
        const consumer_row_count = block_size_x;

        var as: [max_pipeline_stages][block_size_x][block_size_x]f32 addrspace(.shared) = undefined;
        var bs: [max_pipeline_stages][block_size_x][block_size_x]f32 addrspace(.shared) = undefined;
        var produced: [max_pipeline_stages]Mbarrier addrspace(.shared) = undefined;
        var consumed: [max_pipeline_stages]Mbarrier addrspace(.shared) = undefined;

        const SharedStatePipeline = struct {
            head: u32 = 0,
            tail: u32 = 0,
            consumed_phase_parity: u1 = 1,
            produced_phase_parity: u1 = 0,

            inline fn producerAcquire(pipe: *SharedStatePipeline) void {
                while (!consumed[pipe.head].testWaitParity(pipe.consumed_phase_parity)) {}
            }

            inline fn producerCommit(pipe: *SharedStatePipeline) void {
                cp_async.mbarrierArrive(&produced[pipe.head]);
                _ = produced[pipe.head].arrive();
                pipe.head += 1;
                if (pipe.head == max_pipeline_stages) {
                    pipe.head = 0;
                    pipe.consumed_phase_parity ^= 1;
                }
            }

            inline fn consumerWait(pipe: *SharedStatePipeline) void {
                while (!produced[pipe.tail].testWaitParity(pipe.produced_phase_parity)) {}
            }

            inline fn consumerRelease(pipe: *SharedStatePipeline) void {
                _ = consumed[pipe.tail].arrive();
                pipe.tail += 1;
                if (pipe.tail == max_pipeline_stages) {
                    pipe.tail = 0;
                    pipe.produced_phase_parity ^= 1;
                }
            }
        };

        fn matrixMulAsyncCopyMultiStageSharedState(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            var c_sub: f32 = 0.0;

            const a_begin = wa * block_size_x * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size_x;
            const b_begin = block_size_x * bx;
            const b_step = block_size_x * wb;

            if (tx == 0 and ty == 0) {
                const producer_count = (@workGroupSize(1) - consumer_row_count) * @workGroupSize(0);
                const consumer_count = consumer_row_count * @workGroupSize(0);
                inline for (0..max_pipeline_stages) |stage| {
                    Mbarrier.init(&consumed[stage], consumer_count);
                    Mbarrier.init(&produced[stage], producer_count);
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            var pipe: SharedStatePipeline = .{};

            var a_index = a_begin;
            var b_index = b_begin;
            var i: u32 = 0;
            var a_stage = a_begin;
            var b_stage = b_begin;
            var i_stage: u32 = 0;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
                i += 1;
            }) {
                if (ty >= consumer_row_count) {
                    while (a_stage <= a_index + a_step * max_pipeline_stages) : ({
                        a_stage += a_step;
                        b_stage += b_step;
                        i_stage += 1;
                    }) {
                        if (a_stage <= a_end) {
                            const j = i_stage % max_pipeline_stages;
                            const stride_rows = @workGroupSize(1) - consumer_row_count;
                            pipe.producerAcquire();
                            var row_id = ty - consumer_row_count;
                            while (row_id < block_size_x) : (row_id += stride_rows) {
                                cp_async.copy(.cache_all_levels, &as[j][row_id][tx], &a[a_stage + wa * row_id + tx]);
                                cp_async.copy(.cache_all_levels, &bs[j][row_id][tx], &b[b_stage + wb * row_id + tx]);
                            }
                            pipe.producerCommit();
                        }
                    }
                } else {
                    const j = i % max_pipeline_stages;
                    pipe.consumerWait();
                    inline for (0..block_size_x) |k| {
                        c_sub += as[j][ty][k] * bs[j][k][tx];
                    }
                    pipe.consumerRelease();
                }
            }

            if (ty < consumer_row_count) {
                const c_index = wb * block_size_x * by + block_size_x * bx;
                c[c_index + wb * ty + tx] = c_sub;
            }
        }

        comptime {
            @export(&matrixMulAsyncCopyMultiStageSharedState, .{ .name = "matrixMulAsyncCopyMultiStageSharedState" });
        }
    };
}

fn MatrixMulNaive(comptime block_size: u32) type {
    return struct {
        var as: [block_size][block_size]f32 addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 addrspace(.shared) = undefined;

        fn matrixMulNaive(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0;

            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                as[ty][tx] = a[a_index + wa * ty + tx];
                bs[ty][tx] = b[b_index + wb * ty + tx];

                nvptx.barrierSyncAligned(.work_group, null);

                inline for (0..block_size) |k| {
                    c_sub += as[ty][k] * bs[k][tx];
                }

                nvptx.barrierSyncAligned(.work_group, null);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulNaive, .{ .name = "matrixMulNaive" });
        }
    };
}

fn MatrixMulNaiveLargeChunk(comptime block_size: u32) type {
    return struct {
        var as: [block_size][block_size]f32 align(16) addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 align(16) addrspace(.shared) = undefined;

        fn matrixMulNaiveLargeChunk(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: u32,
            wb: u32,
        ) callconv(.kernel) void {
            const bx = @workGroupId(0);
            const by = @workGroupId(1);
            const tx = @workItemId(0);
            const ty = @workItemId(1);

            const t4x = tx * 4;

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0;

            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                if (t4x < block_size) {
                    sharedInt4(&as[ty][t4x]).* = globalInt4(&a[a_index + wa * ty + t4x]).*;
                    sharedInt4(&bs[ty][t4x]).* = globalInt4(&b[a_index + wa * ty + t4x]).*;
                }

                nvptx.barrierSyncAligned(.work_group, null);

                inline for (0..block_size) |k| {
                    c_sub += as[ty][k] * bs[k][tx];
                }

                nvptx.barrierSyncAligned(.work_group, null);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMulNaiveLargeChunk, .{ .name = "matrixMulNaiveLargeChunk" });
        }
    };
}

comptime {
    const block_size = 16;
    _ = MatrixMulAsyncCopyMultiStageLargeChunk(block_size);
    _ = MatrixMulAsyncCopyLargeChunk(block_size);
    _ = MatrixMulAsyncCopyLargeChunkAwBarrier(block_size);
    _ = MatrixMulAsyncCopyMultiStageSharedState(block_size);
    _ = MatrixMulAsyncCopyMultiStage(block_size);
    _ = MatrixMulAsyncCopySingleStage(block_size);
    _ = MatrixMulNaive(block_size);
    _ = MatrixMulNaiveLargeChunk(block_size);
}
