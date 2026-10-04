const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;

pub const Tiling = struct {
    block_m: comptime_int,
    block_n: comptime_int,
    block_k: comptime_int,
    stages: comptime_int,
    warp_m: comptime_int,
    warp_n: comptime_int,
    /// Thread block swizzling: consecutive work groups take `group_m` block rows of one column.
    group_m: comptime_int,
    min_work_groups_per_sm: comptime_int,
};

pub const kernels = .{
    Gemm(.{
        .block_m = 128,
        .block_n = 256,
        .block_k = 64,
        .stages = 2,
        .warp_m = 64,
        .warp_n = 64,
        .group_m = 12,
        .min_work_groups_per_sm = 1,
    }),
};

comptime {
    if (builtin.cpu.arch.isNvptx()) for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
}

const TensorMap = extern struct {
    opaque_bytes: [128]u8 align(64),
};

pub fn Gemm(comptime c: Tiling) type {
    if (c.block_k * @sizeOf(u16) != 64 and c.block_k * @sizeOf(u16) != 128)
        @compileError("the shared layout requires a 64-byte or 128-byte TMA swizzle");
    if (c.group_m == 0) @compileError("the thread block swizzling needs a group_m of at least 1");
    if (c.block_m % c.warp_m != 0 or c.block_n % c.warp_n != 0)
        @compileError("the warp tiles must divide the block tile");
    if (c.warp_m % 16 != 0 or c.warp_n % 16 != 0 or c.block_k % 16 != 0)
        @compileError("the warp tile must contain complete 16x16 matrix loads");
    if (c.stages < 2) @compileError("the TMA pipeline requires at least two shared buffers");
    return struct {
        pub const name = std.fmt.comptimePrint("gemm_bf16_{d}x{d}x{d}_s{d}_w{d}x{d}_g{d}", .{
            c.block_m, c.block_n, c.block_k, c.stages, c.warp_m, c.warp_n, c.group_m,
        });
        pub const block_m = c.block_m;
        pub const block_n = c.block_n;
        pub const block_k = c.block_k;
        pub const group_m = c.group_m;
        pub const threads = c.block_m / c.warp_m * (c.block_n / c.warp_n) * 32;
        pub const swizzle_bytes = c.block_k * @sizeOf(u16);
        pub const dynamic_shared_bytes = @sizeOf(Tiles);
        const Tiles = extern struct {
            a: [c.stages][c.block_m * c.block_k]u16,
            b: [c.stages][c.block_n * c.block_k]u16,
        };
        const Accumulators = [c.warp_m / 16][c.warp_n / 8]@Vector(4, f32);
        const mma_opt: nvptx.matrix.MmaOptions = .{ .shape = .m16n8k16, .accumulator = .f32, .element = .bf16 };

        // Dynamic shared memory has no PTX type, and the tiles exceed the 48 KiB of static memory.
        extern var shared_storage: [0]u8 align(swizzle_bytes * 8) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        // A stage is full after its bulk tensor copies and empty after every warp has read it.
        var full: [c.stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [c.stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        /// Asserts the work group size and dimensions. Assumes that the grid and tensor maps match.
        fn kernel(
            map_a: TensorMap,
            map_b: TensorMap,
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
            if (warp == 0 and nvptx.electSync(.all)) {
                inline for (&full, &empty) |*full_stage, *empty_stage| {
                    full_stage.init(1);
                    empty_stage.init(threads / 32);
                }
                nvptx.fenceProxyAsync(.shared);
                inline for (0..c.stages) |stage| {
                    if (stage * c.block_k < k) load(stage, &map_a, &map_b, m_offset, n_offset, stage * c.block_k);
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            const warp_m = warp / (c.block_n / c.warp_n) * c.warp_m;
            const warp_n = warp % (c.block_n / c.warp_n) * c.warp_n;
            const zero: @Vector(4, f32) = @splat(0);
            var accumulators: Accumulators = @splat(@splat(zero));
            const k_tiles = k / c.block_k;
            var stage: u32 = 0;
            var phase: u1 = 0;
            while (!full[0].tryWaitParity(0)) {}
            var fragments = loadFragments(0, 0, warp_m, warp_n, lane);
            var k_tile: u32 = 0;
            while (k_tile < k_tiles) : (k_tile += 1) {
                // Each warp loads the fragments of the next k step before it multiplies those of
                // the current one. Each warp releases a stage after its last `ldmatrix` from it.
                inline for (0..c.block_k / 16) |k_step| {
                    const current = fragments;
                    if (k_step + 1 < c.block_k / 16) {
                        fragments = loadFragments(stage, k_step + 1, warp_m, warp_n, lane);
                    } else {
                        if (nvptx.electSync(.all)) _ = empty[stage].arrive();
                        if (warp == 0 and k_tile + c.stages < k_tiles and nvptx.electSync(.all)) {
                            while (!empty[stage].tryWaitParity(phase)) {}
                            load(stage, &map_a, &map_b, m_offset, n_offset, (k_tile + c.stages) * c.block_k);
                        }
                        stage += 1;
                        if (stage == c.stages) {
                            stage = 0;
                            phase ^= 1;
                        }
                    }
                    multiply(&accumulators, current);
                    if (k_step + 1 == c.block_k / 16 and k_tile + 1 < k_tiles) {
                        while (!full[stage].tryWaitParity(phase)) {}
                        fragments = loadFragments(stage, 0, warp_m, warp_n, lane);
                    }
                }
            }
            store(out, n, m_offset + warp_m, n_offset + warp_n, lane, &accumulators);
        }

        inline fn load(
            stage: u32,
            map_a: *const TensorMap,
            map_b: *const TensorMap,
            m_offset: u32,
            n_offset: u32,
            k_offset: u32,
        ) void {
            _ = full[stage].arriveExpectTx(@sizeOf(Tiles) / c.stages);
            asm volatile (
                \\cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[a]], [%[map_a], {%[k], %[m]}], [%[bar]];
                \\cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[b]], [%[map_b], {%[k], %[n]}], [%[bar]];
                :
                : [bar] "r" (&full[stage]),
                  [a] "r" (&tiles.a[stage]),
                  [b] "r" (&tiles.b[stage]),
                  [map_a] "l" (map_a),
                  [map_b] "l" (map_b),
                  [k] "r" (k_offset),
                  [m] "r" (m_offset),
                  [n] "r" (n_offset),
                : .{ .memory = true });
        }

        /// Returns the index of `column` of row `row` in a stage. The copy moves 16-byte chunk `x`
        /// of each 128-byte line `y` to chunk `x ^ y % (swizzle_bytes / 16)` (PTX ISA 9.4, 5.5.7).
        inline fn swizzle(row: u32, column: u32) u32 {
            const index = row * c.block_k + column;
            const chunk_size = 16 / @sizeOf(u16);
            return index ^ (index / chunk_size & (swizzle_bytes / 16 - 1) * chunk_size);
        }

        const Fragments = struct {
            a: [c.warp_m / 16]@Vector(4, u32),
            b: [c.warp_n / 8]@Vector(2, u32),
        };

        // PTX ISA 9.4, section 9.7.16.5.8 gives the 8x8 matrices of the A and B registers.
        inline fn loadFragments(stage: u32, comptime k_step: u32, warp_m: u32, warp_n: u32, lane: u32) Fragments {
            const a_tile = &tiles.a[stage];
            const b_tile = &tiles.b[stage];
            var fragments: Fragments = undefined;
            inline for (&fragments.a, 0..) |*fragment, i| {
                const row = warp_m + @as(u32, i) * 16 + lane % 16;
                const column = k_step * 16 + lane / 16 * 8;
                fragment.* = nvptx.ldmatrix(.x4, false, &a_tile[swizzle(row, column)]);
            }
            // One `ldmatrix.x4` loads the two k halves of n rows 0-7 and then those of n rows 8-15,
            // so that each B fragment lands in two adjacent registers.
            inline for (0..c.warp_n / 16) |j| {
                const row = warp_n + @as(u32, j) * 16 + lane % 8 + lane / 16 * 8;
                const column = k_step * 16 + lane / 8 % 2 * 8;
                const pair = nvptx.ldmatrix(.x4, false, &b_tile[swizzle(row, column)]);
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
            inline for (accumulators, 0..) |row_fragments, i| {
                inline for (row_fragments, 0..) |fragment, j| {
                    const row = warp_m + @as(u32, i) * 16 + lane / 4;
                    const column = warp_n + @as(u32, j) * 8 + lane % 4 * 2;
                    const index = @as(u64, row) * n + column;
                    const top: *addrspace(.global) @Vector(2, f32) = @ptrCast(@alignCast(&out[index]));
                    const bottom: *addrspace(.global) @Vector(2, f32) = @ptrCast(@alignCast(&out[index + 8 * @as(u64, n)]));
                    // `st.cs` marks C evict-first to spare A and B in L2 (PTX ISA 9.7.10.1).
                    nvptx.cache.store(.{ .operator = .cache_streaming }, top, @shuffle(f32, fragment, undefined, [2]i32{ 0, 1 }));
                    nvptx.cache.store(.{ .operator = .cache_streaming }, bottom, @shuffle(f32, fragment, undefined, [2]i32{ 2, 3 }));
                }
            }
        }
    };
}
