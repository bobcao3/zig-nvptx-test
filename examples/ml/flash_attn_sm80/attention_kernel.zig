//! A forward pass of FlashAttention-2 (Dao, 2023), with the softmax computed online in base 2.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;

pub const Tiling = struct {
    head_dim: comptime_int,
    /// A work group computes `block_m` query rows. Each warp computes 16 rows.
    block_m: comptime_int,
    block_n: comptime_int,
    stages: comptime_int,
    min_work_groups_per_sm: comptime_int,
};

pub const kernels = .{
    Attention(.{
        .head_dim = 128,
        .block_m = 128,
        .block_n = 64,
        .stages = 2,
        .min_work_groups_per_sm = 1,
    }),
    Attention(.{
        .head_dim = 128,
        .block_m = 128,
        .block_n = 64,
        .stages = 3,
        .min_work_groups_per_sm = 1,
    }),
    Attention(.{
        .head_dim = 128,
        .block_m = 128,
        .block_n = 96,
        .stages = 2,
        .min_work_groups_per_sm = 1,
    }),
    Attention(.{
        .head_dim = 64,
        .block_m = 128,
        .block_n = 64,
        .stages = 2,
        .min_work_groups_per_sm = 2,
    }),
    Attention(.{
        .head_dim = 64,
        .block_m = 64,
        .block_n = 64,
        .stages = 2,
        .min_work_groups_per_sm = 3,
    }),
};

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
    }
}

/// A chunk holds eight bf16 elements. One `cp.async` copy and one `ldmatrix` row move a chunk.
const Chunk = @Vector(4, u32);
const Accumulator = @Vector(4, f32);
/// A row pair holds values of the rows `lane / 4` and `lane / 4 + 8` of the 16 rows of a warp.
const RowPair = @Vector(2, f32);

const mma_options: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k16,
    .accumulator = .f32,
    .element = .bf16,
};

pub fn Attention(comptime c: Tiling) type {
    const row_chunks = c.head_dim / 8;
    const warps = c.block_m / 16;
    const copy_rows = warps * 32 / row_chunks;
    if (c.head_dim != 64 and c.head_dim != 128) @compileError("head_dim must be 64 or 128");
    if (c.block_m % 16 != 0 or c.block_n % 16 != 0)
        @compileError("block_m and block_n must be multiples of 16");
    if (copy_rows % 8 != 0 or c.block_m % copy_rows != 0 or c.block_n % copy_rows != 0)
        @compileError("each work item must copy chunks of the same swizzled column");
    if (c.stages < 2) @compileError("the copy pipeline needs at least two shared buffers");
    return struct {
        pub const name = std.fmt.comptimePrint("flash_attn_d{d}_{d}x{d}_s{d}_o{d}", .{
            c.head_dim,
            c.block_m,
            c.block_n,
            c.stages,
            c.min_work_groups_per_sm,
        });
        pub const head_dim = c.head_dim;
        pub const block_m = c.block_m;
        pub const threads = warps * 32;
        pub const dynamic_shared_bytes = @sizeOf(Tiles);

        /// The Q tile reuses the last stage when it fits, and leaves it before its first refill.
        const query_in_stage = c.block_m <= 2 * c.block_n;
        const Stage = struct {
            k: [c.block_n * row_chunks]Chunk,
            v: [c.block_n * row_chunks]Chunk,
        };
        const Tiles = struct {
            stages: [c.stages]Stage,
            q: if (query_in_stage) void else [c.block_m * row_chunks]Chunk,
        };
        const Query = [c.head_dim / 16]@Vector(4, u32);
        const Scores = [c.block_n / 8]Accumulator;
        const Probabilities = [c.block_n / 16]@Vector(4, u32);
        const Output = [c.head_dim / 8]Accumulator;
        const Rows = struct {
            output: Output,
            max: RowPair,
            sum: RowPair,
        };

        extern var shared_storage: [0]u8 align(128) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        const query_tile: *addrspace(.shared) [c.block_m * row_chunks]Chunk = if (query_in_stage)
            @ptrCast(&tiles.stages[c.stages - 1])
        else
            &tiles.q;

        /// Q, K, V and O have the layout [batch, heads, seq, head_dim]. Work group (x, y) computes
        /// a query block of batch and head pair y. `scale_log2` is the softmax scale times log2(e).
        /// Asserts the work group size and `seq > 0`. Assumes `ceil(seq / block_m)` by y groups.
        fn kernel(
            q: [*]addrspace(.global) const Chunk,
            k: [*]addrspace(.global) const Chunk,
            v: [*]addrspace(.global) const Chunk,
            o: [*]addrspace(.global) u32,
            seq: u32,
            scale_log2: f32,
            causal: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
            .min_work_groups_per_sm = c.min_work_groups_per_sm,
        } }) void {
            @setEvalBranchQuota(20_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            std.debug.assert(@workGroupSize(0) == threads);
            std.debug.assert(seq > 0);
            // The last query blocks of causal attention have the most keys, so they start first.
            const query_block = if (causal != 0)
                nvptx.nctaid(0) - 1 - @workGroupId(0)
            else
                @workGroupId(0);
            const first_query = query_block * c.block_m;
            if (first_query >= seq) return;
            const head_row = @as(u64, @workGroupId(1)) * seq;
            const key_end = if (causal != 0) @min(first_query + c.block_m, seq) else seq;
            const tile_count = (key_end + c.block_n - 1) / c.block_n;
            const unmasked_tiles = (if (causal != 0) first_query + 1 else seq) / c.block_n;
            const copy: Copy = .{
                .slot = swizzle(tid / row_chunks, tid % row_chunks),
                .row = tid / row_chunks,
                .column = head_row * row_chunks + tid % row_chunks,
                .seq = seq,
            };
            copy.rows(c.block_m, query_tile, q, first_query, true);
            inline for (0..c.stages - 1) |stage| {
                if (stage < tile_count) copy.tile(k, v, stage, stage);
                cp_async.commitGroup();
            }
            cp_async.waitGroup(c.stages - 2);
            nvptx.barrierSyncAligned(.work_group, null);
            const query = loadQuery(warp, lane);
            var rows: Rows = .{
                .output = @splat(@splat(0)),
                .max = @splat(-std.math.inf(f32)),
                .sum = @splat(0),
            };
            const first_row = first_query + warp * 16 + lane / 4;
            const seqs: @Vector(2, u32) = @splat(seq);
            const row_limits: @Vector(2, u32) = if (causal != 0)
                @min(@Vector(2, u32){ first_row + 1, first_row + 9 }, seqs)
            else
                seqs;
            var stage: u32 = 0;
            var tile: u32 = 0;
            while (tile < unmasked_tiles) : (tile += 1) {
                copy.next(k, v, stage, tile, tile_count);
                const scores = computeScores(query, stage, lane);
                rows = attend(rows, scores, stage, lane, scale_log2, false, undefined);
                stage = if (stage + 1 == c.stages) 0 else stage + 1;
            }
            while (tile < tile_count) : (tile += 1) {
                copy.next(k, v, stage, tile, tile_count);
                const scores = computeScores(query, stage, lane);
                const mask: Mask = .{ .first_key = tile * c.block_n, .row_limits = row_limits };
                rows = attend(rows, scores, stage, lane, scale_log2, true, mask);
                stage = if (stage + 1 == c.stages) 0 else stage + 1;
            }
            store(o, head_row, first_row, seq, lane, rows);
        }

        /// Each work item copies one chunk column of a tile. Its rows lie `copy_rows` apart.
        const Copy = struct {
            slot: u32,
            row: u32,
            column: u64,
            seq: u32,

            /// When `bounded`, rows at or after `seq` copy row `seq - 1`, inside the tensor.
            inline fn rows(
                copy: Copy,
                comptime count: u32,
                destination: *addrspace(.shared) [count * row_chunks]Chunk,
                source: [*]addrspace(.global) const Chunk,
                first_row: u32,
                comptime bounded: bool,
            ) void {
                inline for (0..count / copy_rows) |pass| {
                    const row = first_row + copy.row + @as(u32, pass) * copy_rows;
                    const source_row = if (bounded) @min(row, copy.seq - 1) else row;
                    const slot = copy.slot + @as(u32, pass) * copy_rows * row_chunks;
                    const chunk = copy.column + @as(u64, source_row) * row_chunks;
                    cp_async.copy(.cache_global_level, &destination[slot], &source[chunk]);
                }
            }

            inline fn tile(
                copy: Copy,
                k: [*]addrspace(.global) const Chunk,
                v: [*]addrspace(.global) const Chunk,
                stage: u32,
                index: u32,
            ) void {
                const first_key = index * c.block_n;
                if (first_key + c.block_n <= copy.seq) {
                    copy.rows(c.block_n, &tiles.stages[stage].k, k, first_key, false);
                    copy.rows(c.block_n, &tiles.stages[stage].v, v, first_key, false);
                } else {
                    copy.rows(c.block_n, &tiles.stages[stage].k, k, first_key, true);
                    copy.rows(c.block_n, &tiles.stages[stage].v, v, first_key, true);
                }
            }

            /// Waits for tile `current` in `stage` and refills the stage of the previous tile. The
            /// barrier shares the copies and orders the refill after the reads of that stage.
            inline fn next(
                copy: Copy,
                k: [*]addrspace(.global) const Chunk,
                v: [*]addrspace(.global) const Chunk,
                stage: u32,
                current: u32,
                tile_count: u32,
            ) void {
                cp_async.waitGroup(c.stages - 2);
                nvptx.barrierSyncAligned(.work_group, null);
                const refill = current + c.stages - 1;
                const refill_stage = if (stage == 0) c.stages - 1 else stage - 1;
                if (refill < tile_count) copy.tile(k, v, refill_stage, refill);
                cp_async.commitGroup();
            }
        };

        /// Returns the index of chunk `column` of `row` in a tile. The XOR with `row % 8` gives the
        /// eight rows of an `ldmatrix` matrix different groups of four banks.
        inline fn swizzle(row: u32, column: u32) u32 {
            return row * row_chunks + (column ^ row % 8);
        }

        /// Returns a pointer to chunk `column` of `row` in `tile`. Assumes `row % 8 == lane % 8`.
        inline fn matrixChunk(
            tile: anytype,
            row: u32,
            column: u32,
            lane: u32,
        ) *addrspace(.shared) Chunk {
            return &tile[row * row_chunks + (column ^ lane % 8)];
        }

        // PTX ISA 9.4 sections 9.7.16.5.8 and 9.7.16.5.15 give the `mma` and `ldmatrix` layouts.
        inline fn loadQuery(warp: u32, lane: u32) Query {
            var query: Query = undefined;
            inline for (&query, 0..) |*fragment, k_step| {
                const row = warp * 16 + lane % 16;
                const column = @as(u32, k_step) * 2 + lane / 16;
                const chunk = matrixChunk(query_tile, row, column, lane);
                fragment.* = nvptx.ldmatrix(.x4, false, chunk);
            }
            return query;
        }

        inline fn computeScores(query: Query, stage: u32, lane: u32) Scores {
            const tile = &tiles.stages[stage].k;
            var scores: Scores = @splat(@splat(0));
            inline for (query, 0..) |a, k_step| {
                // One `ldmatrix.x4` loads dimensions 0-15 of keys 0-7, then of keys 8-15.
                inline for (0..c.block_n / 16) |j| {
                    const row = @as(u32, j) * 16 + lane % 8 + lane / 16 * 8;
                    const column = @as(u32, k_step) * 2 + lane / 8 % 2;
                    const chunk = matrixChunk(tile, row, column, lane);
                    const b = nvptx.ldmatrix(.x4, false, chunk);
                    const low: @Vector(2, u32) = .{ b[0], b[1] };
                    const high: @Vector(2, u32) = .{ b[2], b[3] };
                    scores[2 * j] = nvptx.mmaSync(mma_options, a, low, scores[2 * j]);
                    scores[2 * j + 1] = nvptx.mmaSync(mma_options, a, high, scores[2 * j + 1]);
                }
            }
            return scores;
        }

        const Mask = struct {
            first_key: u32,
            row_limits: @Vector(2, u32),
        };

        /// Returns `rows` after the online softmax of `scores` and the products with the V tile.
        inline fn attend(
            rows: Rows,
            scores_in: Scores,
            stage: u32,
            lane: u32,
            scale_log2: f32,
            comptime masked: bool,
            mask: Mask,
        ) Rows {
            var scores = scores_in;
            if (masked) {
                const limits: @Vector(4, u32) = .{
                    mask.row_limits[0],
                    mask.row_limits[0],
                    mask.row_limits[1],
                    mask.row_limits[1],
                };
                inline for (&scores, 0..) |*fragment, n_tile| {
                    const key = mask.first_key + @as(u32, n_tile) * 8 + lane % 4 * 2;
                    const keys: @Vector(4, u32) = .{ key, key + 1, key, key + 1 };
                    const minus_infinity: Accumulator = @splat(-std.math.inf(f32));
                    fragment.* = @select(f32, keys < limits, fragment.*, minus_infinity);
                }
            }
            var max = rows.max;
            inline for (scores) |fragment| {
                const even: RowPair = .{ fragment[0], fragment[2] };
                const odd: RowPair = .{ fragment[1], fragment[3] };
                max = @max(max, @max(even, odd));
            }
            max = @max(max, shuffleXor(max, 1));
            max = @max(max, shuffleXor(max, 2));
            const scale: RowPair = @splat(scale_log2);
            const scaled_max = max * scale;
            const correction = exp2(rows.max * scale - scaled_max);
            const factors: Accumulator = .{
                correction[0],
                correction[0],
                correction[1],
                correction[1],
            };
            var output = rows.output;
            inline for (&output) |*fragment| fragment.* *= factors;
            var sum = rows.sum * correction;
            const offsets: Accumulator = .{
                scaled_max[0],
                scaled_max[0],
                scaled_max[1],
                scaled_max[1],
            };
            const scale4: Accumulator = @splat(scale_log2);
            var probabilities: Probabilities = undefined;
            inline for (&probabilities, 0..) |*fragment, k_step| {
                const left = exp2(@mulAdd(Accumulator, scores[2 * k_step], scale4, -offsets));
                const right = exp2(@mulAdd(Accumulator, scores[2 * k_step + 1], scale4, -offsets));
                sum += RowPair{ left[0] + left[1], left[2] + left[3] };
                sum += RowPair{ right[0] + right[1], right[2] + right[3] };
                // Two accumulator tiles of 8 keys have the A layout of one tile of 16 keys.
                fragment.* = .{
                    @backingInt(nvptx.Bf16x2.fromF32(left[0], left[1])),
                    @backingInt(nvptx.Bf16x2.fromF32(left[2], left[3])),
                    @backingInt(nvptx.Bf16x2.fromF32(right[0], right[1])),
                    @backingInt(nvptx.Bf16x2.fromF32(right[2], right[3])),
                };
            }
            const tile = &tiles.stages[stage].v;
            inline for (probabilities, 0..) |a, k_step| {
                // One `ldmatrix.x4.trans` loads keys 0-15 of dimensions 0-7, then of 8-15.
                inline for (0..c.head_dim / 16) |j| {
                    const row = @as(u32, k_step) * 16 + lane % 8 + lane / 8 % 2 * 8;
                    const column = @as(u32, j) * 2 + lane / 16;
                    const chunk = matrixChunk(tile, row, column, lane);
                    const b = nvptx.ldmatrix(.x4, true, chunk);
                    const low: @Vector(2, u32) = .{ b[0], b[1] };
                    const high: @Vector(2, u32) = .{ b[2], b[3] };
                    output[2 * j] = nvptx.mmaSync(mma_options, a, low, output[2 * j]);
                    output[2 * j + 1] = nvptx.mmaSync(mma_options, a, high, output[2 * j + 1]);
                }
            }
            return .{ .output = output, .max = max, .sum = sum };
        }

        inline fn store(
            o: [*]addrspace(.global) u32,
            head_row: u64,
            first_row: u32,
            seq: u32,
            lane: u32,
            rows: Rows,
        ) void {
            var sum = rows.sum;
            sum += shuffleXor(sum, 1);
            sum += shuffleXor(sum, 2);
            const inverse = @as(RowPair, @splat(1)) / sum;
            const row_words = c.head_dim / 2;
            inline for (rows.output, 0..) |fragment, n_tile| {
                const word = @as(u32, n_tile) * 4 + lane % 4;
                const top = (head_row + first_row) * row_words + word;
                if (first_row < seq) o[top] = @backingInt(nvptx.Bf16x2.fromF32(
                    fragment[0] * inverse[0],
                    fragment[1] * inverse[0],
                ));
                if (first_row + 8 < seq) o[top + 8 * row_words] = @backingInt(nvptx.Bf16x2.fromF32(
                    fragment[2] * inverse[1],
                    fragment[3] * inverse[1],
                ));
            }
        }
    };
}

inline fn shuffleXor(x: RowPair, comptime mask: u32) RowPair {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return .{
        nvptx.shflSync(.bfly, f32, x[0], mask, clamp, .all),
        nvptx.shflSync(.bfly, f32, x[1], mask, clamp, .all),
    };
}

inline fn exp2(x: anytype) @TypeOf(x) {
    var result: @TypeOf(x) = undefined;
    inline for (0..@typeInfo(@TypeOf(x)).vector.len) |i| {
        result[i] = nvptx.math.ex2ApproxF32(true, x[i]);
    }
    return result;
}
