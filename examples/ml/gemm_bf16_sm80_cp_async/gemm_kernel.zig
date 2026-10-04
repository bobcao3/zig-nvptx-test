const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;

pub const Tiling = struct {
    block_m: comptime_int,
    block_n: comptime_int,
    block_k: comptime_int,
    stages: comptime_int,
    warp_m: comptime_int,
    warp_n: comptime_int,
    /// Consecutive work groups take up to `group_m` block rows of one block column for L2 reuse.
    group_m: comptime_int,
    min_work_groups_per_sm: comptime_int,
};

pub const Kernel = Gemm(.{
    .block_m = 256,
    .block_n = 128,
    .block_k = 64,
    .stages = 2,
    .warp_m = 64,
    .warp_n = 64,
    .group_m = 6,
    .min_work_groups_per_sm = 1,
});

comptime {
    if (builtin.cpu.arch.isNvptx()) @export(&Kernel.kernel, .{ .name = Kernel.name });
}

/// A chunk holds eight bf16 elements, the size of one `cp.async` copy and of one `ldmatrix` row.
const Chunk = @Vector(4, u32);

pub fn Gemm(comptime c: Tiling) type {
    const row_chunks = c.block_k / 8;
    const warps = c.block_m / c.warp_m * (c.block_n / c.warp_n);
    if (row_chunks != 4 and row_chunks != 8)
        @compileError("the shared layout needs a block_k of 32 or 64");
    if (c.group_m == 0) @compileError("the thread block swizzling needs a group_m of at least 1");
    if (c.block_m % c.warp_m != 0 or c.block_n % c.warp_n != 0)
        @compileError("the warp tiles must divide the block tile");
    if (c.warp_m % 16 != 0 or c.warp_n % 16 != 0)
        @compileError("the warp tile must contain complete 16x16 matrix loads");
    if (warps * 32 / row_chunks % 8 != 0 or c.block_m * row_chunks % (warps * 32) != 0 or
        c.block_n * row_chunks % (warps * 32) != 0)
        @compileError("each work item must copy chunks of the same swizzled column");
    if (c.stages < 2) @compileError("the copy pipeline needs at least two shared buffers");
    return struct {
        pub const name = std.fmt.comptimePrint("gemm_bf16_{d}x{d}x{d}_s{d}_w{d}x{d}_g{d}_o{d}", .{
            c.block_m,
            c.block_n,
            c.block_k,
            c.stages,
            c.warp_m,
            c.warp_n,
            c.group_m,
            c.min_work_groups_per_sm,
        });
        pub const block_m = c.block_m;
        pub const block_n = c.block_n;
        pub const block_k = c.block_k;
        pub const group_m = c.group_m;
        pub const threads = warps * 32;
        pub const dynamic_shared_bytes = @sizeOf(Tiles);
        const Tiles = struct {
            a: [c.stages][c.block_m * row_chunks]Chunk,
            b: [c.stages][c.block_n * row_chunks]Chunk,
        };
        const Accumulators = [c.warp_m / 16][c.warp_n / 8]@Vector(4, f32);
        const mma_opt: nvptx.matrix.MmaOptions = .{
            .shape = .m16n8k16,
            .accumulator = .f32,
            .element = .bf16,
        };
        const k_steps = c.block_k / 16;
        const copy_rows = threads / row_chunks;

        extern var shared_storage: [0]u8 align(128) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);

        /// Computes f32 C = A B^T. Asserts tile-multiple sizes, k > 0 and `threads` work items.
        fn kernel(
            a: [*]addrspace(.global) const Chunk,
            b: [*]addrspace(.global) const Chunk,
            out: [*]addrspace(.global) f32,
            m: u32,
            n: u32,
            k: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
            .min_work_groups_per_sm = c.min_work_groups_per_sm,
        } }) void {
            @setEvalBranchQuota(10_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            const tiles_m = m / c.block_m;
            const tiles_n = n / c.block_n;
            const tile = @workGroupId(1) * tiles_n + @workGroupId(0);
            const group_tiles = c.group_m * tiles_n;
            const first_m = tile / group_tiles * c.group_m;
            const rows_in_group = @min(c.group_m, tiles_m - first_m);
            const tile_in_group = tile % group_tiles;
            const m_offset = (first_m + tile_in_group % rows_in_group) * c.block_m;
            const n_offset = (tile_in_group / rows_in_group) * c.block_n;
            if (m_offset >= m or n_offset >= n) return;
            std.debug.assert(@workGroupSize(0) == threads);
            std.debug.assert(m % c.block_m == 0 and n % c.block_n == 0);
            std.debug.assert(k > 0 and k % c.block_k == 0);
            const row_stride: u64 = k / 8;
            const copy: Copy = .{
                .slot = swizzle(tid / row_chunks, tid % row_chunks),
                .a = a + ((m_offset + tid / row_chunks) * row_stride + tid % row_chunks),
                .b = b + ((n_offset + tid / row_chunks) * row_stride + tid % row_chunks),
                .pass_stride = copy_rows * row_stride,
            };
            const k_tiles = k / c.block_k;
            inline for (0..c.stages) |stage| {
                if (stage < k_tiles) copy.tile(stage, stage);
                cp_async.commitGroup();
            }
            const warp_m = warp / (c.block_n / c.warp_n) * c.warp_m;
            const warp_n = warp % (c.block_n / c.warp_n) * c.warp_n;
            const zero: @Vector(4, f32) = @splat(0);
            var accumulators: Accumulators = @splat(@splat(zero));
            cp_async.waitGroup(c.stages - 1);
            nvptx.barrierSyncAligned(.work_group, null);
            var fragments = loadFragments(0, 0, warp_m, warp_n, lane);
            var stage: u32 = 0;
            var k_tile: u32 = 0;
            while (k_tile < k_tiles) : (k_tile += 1) {
                inline for (0..k_steps) |k_step| {
                    const current = fragments;
                    if (k_step + 1 < k_steps) {
                        fragments = loadFragments(stage, k_step + 1, warp_m, warp_n, lane);
                    } else {
                        // The first fragments of the next stage load without a branch. In the
                        // last k tile, no copy writes that stage, and nothing uses the loads.
                        const next = if (stage + 1 == c.stages) 0 else stage + 1;
                        fragments = loadFragments(next, 0, warp_m, warp_n, lane);
                        if (k_tile + c.stages < k_tiles) copy.tile(stage, k_tile + c.stages);
                        cp_async.commitGroup();
                        stage = next;
                    }
                    multiply(&accumulators, current);
                    // Each work item waits for its copies of the next stage. The barrier shares
                    // them and keeps the refill of `stage` after the last `ldmatrix` of every warp.
                    if (k_step + 2 == k_steps and k_tile + 1 < k_tiles) {
                        cp_async.waitGroup(c.stages - 2);
                        nvptx.barrierSyncAligned(.work_group, null);
                    }
                }
            }
            store(out, n, m_offset + warp_m, n_offset + warp_n, lane, &accumulators);
        }

        /// A work item copies one chunk column of a tile, in rows that lie `copy_rows` apart.
        const Copy = struct {
            slot: u32,
            a: [*]addrspace(.global) const Chunk,
            b: [*]addrspace(.global) const Chunk,
            pass_stride: u64,

            inline fn tile(copy: Copy, stage: u32, k_tile: u32) void {
                const k_offset = k_tile * row_chunks;
                inline for (0..c.block_m / copy_rows) |pass| {
                    const slot = copy.slot + pass * copy_rows * row_chunks;
                    const source = &copy.a[pass * copy.pass_stride + k_offset];
                    cp_async.copy(.cache_global_level, &tiles.a[stage][slot], source);
                }
                inline for (0..c.block_n / copy_rows) |pass| {
                    const slot = copy.slot + pass * copy_rows * row_chunks;
                    const source = &copy.b[pass * copy.pass_stride + k_offset];
                    cp_async.copy(.cache_global_level, &tiles.b[stage][slot], source);
                }
            }
        };

        /// Returns the slot of chunk `column` of row `row`. Chunk `x` of each 128-byte line `y`
        /// moves to chunk `x ^ y % row_chunks`, so that an `ldmatrix` matrix uses all 32 banks.
        inline fn swizzle(row: u32, column: u32) u32 {
            const slot = row * row_chunks + column;
            return slot ^ (slot / 8 % row_chunks);
        }

        const Fragments = struct {
            a: [c.warp_m / 16]@Vector(4, u32),
            b: [c.warp_n / 8]@Vector(2, u32),
        };

        // PTX ISA 9.4, section 9.7.16.5.8 gives the 8x8 matrices of the A and B registers.
        inline fn loadFragments(
            stage: u32,
            comptime k_step: u32,
            warp_m: u32,
            warp_n: u32,
            lane: u32,
        ) Fragments {
            var fragments: Fragments = undefined;
            inline for (&fragments.a, 0..) |*fragment, i| {
                const row = warp_m + @as(u32, i) * 16 + lane % 16;
                const column = k_step * 2 + lane / 16;
                fragment.* = nvptx.ldmatrix(.x4, false, &tiles.a[stage][swizzle(row, column)]);
            }
            // One `ldmatrix.x4` loads the two k halves of n rows 0-7 and then those of n rows 8-15.
            inline for (0..c.warp_n / 16) |j| {
                const row = warp_n + @as(u32, j) * 16 + lane % 8 + lane / 16 * 8;
                const column = k_step * 2 + lane / 8 % 2;
                const pair = nvptx.ldmatrix(.x4, false, &tiles.b[stage][swizzle(row, column)]);
                fragments.b[j * 2] = .{ pair[0], pair[1] };
                fragments.b[j * 2 + 1] = .{ pair[2], pair[3] };
            }
            return fragments;
        }

        inline fn multiply(accumulators: *Accumulators, fragments: Fragments) void {
            inline for (accumulators, fragments.a) |*row, a_fragment| {
                inline for (row, fragments.b) |*fragment, b_fragment| {
                    fragment.* = nvptx.mmaSync(mma_opt, a_fragment, b_fragment, fragment.*);
                }
            }
        }

        inline fn store(
            out: [*]addrspace(.global) f32,
            n: u32,
            warp_m: u32,
            warp_n: u32,
            lane: u32,
            accumulators: *const Accumulators,
        ) void {
            const PairPointer = *addrspace(.global) @Vector(2, f32);
            const streaming: nvptx.cache.StoreCaching = .{ .operator = .cache_streaming };
            inline for (accumulators, 0..) |row_fragments, i| {
                inline for (row_fragments, 0..) |fragment, j| {
                    const row = warp_m + @as(u32, i) * 16 + lane / 4;
                    const column = warp_n + @as(u32, j) * 8 + lane % 4 * 2;
                    const top_index = @as(u64, row) * n + column;
                    const top: PairPointer = @ptrCast(@alignCast(&out[top_index]));
                    const bottom_index = top_index + 8 * @as(u64, n);
                    const bottom: PairPointer = @ptrCast(@alignCast(&out[bottom_index]));
                    const top_value = @shuffle(f32, fragment, undefined, [2]i32{ 0, 1 });
                    const bottom_value = @shuffle(f32, fragment, undefined, [2]i32{ 2, 3 });
                    nvptx.cache.store(streaming, top, top_value);
                    nvptx.cache.store(streaming, bottom, bottom_value);
                }
            }
        }
    };
}
