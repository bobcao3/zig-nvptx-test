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

//! High quality DXT compression using CUDA. This sample shows how to implement an existing
//! computationally intensive CPU compression algorithm in parallel on the GPU.

const std = @import("std");
const math = @import("CudaMath.zig");
const cvt = std.nvptx.math.cvt;

const F3 = math.F3;
const Tile = math.Tile;
const splat3 = math.splat3;
const dot = math.dot;

const num_threads = 64; // The number of work items per work group.

const flt_max = std.math.floatMax(f32);

const k_color_metric: F3 = .{ 1.0, 1.0, 1.0 };

const alpha_table4 = [4]f32{ 9.0, 0.0, 6.0, 3.0 };
const alpha_table3 = [4]f32{ 4.0, 0.0, 2.0, 2.0 };
const prods4 = [4]i32{ 0x090000, 0x000900, 0x040102, 0x010402 };
const prods3 = [4]i32{ 0x040000, 0x000400, 0x040101, 0x010401 };

const use_tables = true;

var colors: [16]F3 addrspace(.shared) = undefined;
var sums: [16]F3 addrspace(.shared) = undefined;
var xrefs: [16]i32 addrspace(.shared) = undefined;
var errors: [num_threads]f32 addrspace(.shared) = undefined;
var dps: [16]f32 addrspace(.shared) = undefined;
var s_permutations: [160]u32 addrspace(.shared) = undefined;
var indices: [num_threads]i32 addrspace(.shared) = undefined;

inline fn sortColors(values: *addrspace(.shared) const [16]f32, ranks: *addrspace(.shared) [16]i32, tile: Tile) void {
    const tid = @workItemId(0);

    var rank: i32 = 0;

    inline for (0..16) |i| {
        rank += @intFromBool(values[i] < values[tid]);
    }

    ranks[tid] = rank;

    tile.sync();

    // Resolve elements with the same index.
    for (0..15) |i| {
        if (tid > i and ranks[tid] == ranks[i]) {
            ranks[tid] += 1;
        }
        tile.sync();
    }
}

inline fn loadColorBlock(image: [*]addrspace(.global) const u32, block_offset: i32) void {
    const bid: u32 = @intCast(@as(i32, @intCast(@workGroupId(0))) + block_offset);
    const idx = @workItemId(0);

    // The tile that holds the first 16 work items of the work group.
    const tile: Tile = .{ .member_mask = @fromBackingInt(0x0000ffff) };

    if (idx < 16) {
        // Read the color and copy it to shared memory.
        const c = image[bid * 16 + idx];

        colors[idx] = .{
            @as(f32, @floatFromInt((c >> 0) & 0xFF)) * (1.0 / 255.0),
            @as(f32, @floatFromInt((c >> 8) & 0xFF)) * (1.0 / 255.0),
            @as(f32, @floatFromInt((c >> 16) & 0xFF)) * (1.0 / 255.0),
        };

        tile.sync();
        // Sort the colors along the best fit line.
        math.colorSums(&colors, &sums, tile);

        tile.sync();

        const axis = math.bestFitLine(&colors, sums[0], tile);

        tile.sync();

        dps[idx] = dot(colors[idx], axis);

        tile.sync();

        sortColors(&dps, &xrefs, tile);

        tile.sync();

        const tmp = colors[idx];

        tile.sync();

        colors[@intCast(xrefs[idx])] = tmp;
    }
}

/// Rounds the color to RGB565 and expands it. The packed color goes to `w`.
inline fn roundAndExpand(v: F3, w: *u16) F3 {
    const rounded: F3 = .{
        cvt(f32, f32, cvt(f32, f32, v[0], .{ .saturate = true }) * 31.0, .{ .rounding = .nearest_even }),
        cvt(f32, f32, cvt(f32, f32, v[1], .{ .saturate = true }) * 63.0, .{ .rounding = .nearest_even }),
        cvt(f32, f32, cvt(f32, f32, v[2], .{ .saturate = true }) * 31.0, .{ .rounding = .nearest_even }),
    };

    const red: u16 = @intFromFloat(rounded[0]);
    const green: u16 = @intFromFloat(rounded[1]);
    const blue: u16 = @intFromFloat(rounded[2]);
    w.* = (red << 11) | (green << 5) | blue;
    // An approximation of the integer bit expansion.
    return rounded * F3{ 0.03227752766457, 0.01583151765563, 0.03227752766457 };
}

/// The sums that the least squares fit of the two endpoints needs.
const FitSums = struct {
    alpha2_sum: f32,
    beta2_sum: f32,
    alphabeta_sum: f32,
    alphax_sum: F3,
    betax_sum: F3,
};

/// Computes the error of the least squares fit and rounds the endpoints `start` and `end`.
inline fn fitEndpoints(sums_of_fit: FitSums, start: *u16, end: *u16) F3 {
    const alpha2_sum = sums_of_fit.alpha2_sum;
    const beta2_sum = sums_of_fit.beta2_sum;
    const alphabeta_sum = sums_of_fit.alphabeta_sum;
    const alphax_sum = sums_of_fit.alphax_sum;
    const betax_sum = sums_of_fit.betax_sum;

    // `alpha2_sum`, `beta2_sum`, `alphabeta_sum` and `factor` could be precomputed for each
    // permutation, but it is faster to recompute them.
    const factor = 1.0 / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    var a = (alphax_sum * splat3(beta2_sum) - betax_sum * splat3(alphabeta_sum)) * splat3(factor);
    var b = (betax_sum * splat3(alpha2_sum) - alphax_sum * splat3(alphabeta_sum)) * splat3(factor);

    // Round `a` and `b` to the closest 5-6-5 color and expand.
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // Compute the error.
    return a * a * splat3(alpha2_sum) + b * b * splat3(beta2_sum) +
        splat3(2.0) * (a * b * splat3(alphabeta_sum) - a * alphax_sum - b * betax_sum);
}

inline fn evalPermutation4(
    permutation: u32,
    start: *u16,
    end: *u16,
    color_sum: F3,
) f32 {
    // Compute the endpoints using least squares.
    const fit: FitSums = if (use_tables) fit: {
        var alphax_sum: F3 = .{ 0.0, 0.0, 0.0 };

        var akku: i32 = 0;

        // Compute alpha and beta for this permutation.
        for (0..16) |i| {
            const bits = permutation >> @intCast(2 * i);

            alphax_sum += splat3(alpha_table4[bits & 3]) * colors[i];
            akku += prods4[bits & 3];
        }

        break :fit .{
            .alpha2_sum = @floatFromInt(akku >> 16),
            .beta2_sum = @floatFromInt((akku >> 8) & 0xff),
            .alphabeta_sum = @floatFromInt((akku >> 0) & 0xff),
            .alphax_sum = alphax_sum,
            .betax_sum = (splat3(9.0) * color_sum) - alphax_sum,
        };
    } else fit: {
        var alpha2_sum: f32 = 0.0;
        var beta2_sum: f32 = 0.0;
        var alphabeta_sum: f32 = 0.0;
        var alphax_sum: F3 = .{ 0.0, 0.0, 0.0 };

        for (0..16) |i| {
            const bits = permutation >> @intCast(2 * i);

            var beta: f32 = @floatFromInt(bits & 1);

            if (bits & 2 != 0) {
                beta = (1 + beta) * (1.0 / 3.0);
            }

            const alpha = 1.0 - beta;

            alpha2_sum += alpha * alpha;
            beta2_sum += beta * beta;
            alphabeta_sum += alpha * beta;
            alphax_sum += splat3(alpha) * colors[i];
        }

        break :fit .{
            .alpha2_sum = alpha2_sum,
            .beta2_sum = beta2_sum,
            .alphabeta_sum = alphabeta_sum,
            .alphax_sum = alphax_sum,
            .betax_sum = color_sum - alphax_sum,
        };
    };

    const e = fitEndpoints(fit, start, end);

    return 0.111111111111 * dot(e, k_color_metric);
}

inline fn evalPermutation3(
    permutation: u32,
    start: *u16,
    end: *u16,
    color_sum: F3,
) f32 {
    // Compute the endpoints using least squares.
    const fit: FitSums = if (use_tables) fit: {
        var alphax_sum: F3 = .{ 0.0, 0.0, 0.0 };

        var akku: i32 = 0;

        // Compute alpha and beta for this permutation.
        for (0..16) |i| {
            const bits = permutation >> @intCast(2 * i);

            alphax_sum += splat3(alpha_table3[bits & 3]) * colors[i];
            akku += prods3[bits & 3];
        }

        break :fit .{
            .alpha2_sum = @floatFromInt(akku >> 16),
            .beta2_sum = @floatFromInt((akku >> 8) & 0xff),
            .alphabeta_sum = @floatFromInt((akku >> 0) & 0xff),
            .alphax_sum = alphax_sum,
            .betax_sum = (splat3(4.0) * color_sum) - alphax_sum,
        };
    } else fit: {
        var alpha2_sum: f32 = 0.0;
        var beta2_sum: f32 = 0.0;
        var alphabeta_sum: f32 = 0.0;
        var alphax_sum: F3 = .{ 0.0, 0.0, 0.0 };

        for (0..16) |i| {
            const bits = permutation >> @intCast(2 * i);

            var beta: f32 = @floatFromInt(bits & 1);

            if (bits & 2 != 0) {
                beta = 0.5;
            }

            const alpha = 1.0 - beta;

            alpha2_sum += alpha * alpha;
            beta2_sum += beta * beta;
            alphabeta_sum += alpha * beta;
            alphax_sum += splat3(alpha) * colors[i];
        }

        break :fit .{
            .alpha2_sum = alpha2_sum,
            .beta2_sum = beta2_sum,
            .alphabeta_sum = alphabeta_sum,
            .alphax_sum = alphax_sum,
            .betax_sum = color_sum - alphax_sum,
        };
    };

    const e = fitEndpoints(fit, start, end);

    return 0.25 * dot(e, k_color_metric);
}

inline fn evalAllPermutations(
    permutations: [*]addrspace(.global) const u32,
    best_start: *u16,
    best_end: *u16,
    best_permutation: *u32,
    color_sum: F3,
) void {
    const idx = @workItemId(0);

    var best_error: f32 = flt_max;

    for (0..16) |i| {
        const pidx = idx + num_threads * i;

        if (pidx >= 992) {
            break;
        }

        var start: u16 = undefined;
        var end: u16 = undefined;
        const permutation = permutations[pidx];

        if (pidx < 160) {
            s_permutations[pidx] = permutation;
        }

        const current_error = evalPermutation4(permutation, &start, &end, color_sum);

        if (current_error < best_error) {
            best_error = current_error;
            best_permutation.* = permutation;
            best_start.* = start;
            best_end.* = end;
        }
    }

    if (best_start.* < best_end.*) {
        std.mem.swap(u16, best_end, best_start);
        best_permutation.* ^= 0x55555555; // Flip the indices.
    }

    std.nvptx.barrierSyncAligned(.work_group, null); // Sync here to ensure that s_permutations is valid going forward.

    for (0..3) |i| {
        const pidx = idx + num_threads * i;

        if (pidx >= 160) {
            break;
        }

        var start: u16 = undefined;
        var end: u16 = undefined;
        const permutation = s_permutations[pidx];
        const current_error = evalPermutation3(permutation, &start, &end, color_sum);

        if (current_error < best_error) {
            best_error = current_error;
            best_permutation.* = permutation;
            best_start.* = start;
            best_end.* = end;

            if (best_start.* > best_end.*) {
                std.mem.swap(u16, best_end, best_start);
                best_permutation.* ^= (~best_permutation.* >> 1) & 0x55555555; // Flip the indices.
            }
        }
    }

    errors[idx] = best_error;
}

/// Finds the index with the minimum error.
inline fn findMinError() i32 {
    const idx = @workItemId(0);
    indices[idx] = @intCast(idx);

    std.nvptx.barrierSyncAligned(.work_group, null);

    var d: u32 = num_threads / 2;
    while (d > 0) : (d >>= 1) {
        const err0 = errors[idx];
        const err1 = if (idx + d < num_threads) errors[idx + d] else flt_max;
        const index1 = if (idx + d < num_threads) indices[idx + d] else 0;

        std.nvptx.barrierSyncAligned(.work_group, null);

        if (err1 < err0) {
            errors[idx] = err1;
            indices[idx] = index1;
        }

        std.nvptx.barrierSyncAligned(.work_group, null);
    }

    return indices[0];
}

const Uint2 = extern struct { x: u32, y: u32 };

inline fn saveBlockDxt1(
    start: u16,
    end: u16,
    permutation_in: u32,
    result: [*]addrspace(.global) Uint2,
    block_offset: i32,
) void {
    const bid: u32 = @intCast(@as(i32, @intCast(@workGroupId(0))) + block_offset);

    const permutation = if (start == end) 0 else permutation_in;

    // Reorder the permutation.
    var palette_indices: u32 = 0;

    inline for (0..16) |i| {
        const ref: u5 = @intCast(xrefs[i]);
        palette_indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
    }

    // Write the endpoints.
    result[bid].x = (@as(u32, end) << 16) | start;

    // Write the palette indices.
    result[bid].y = palette_indices;
}

/// Compresses a color block.
export fn compress(
    permutations: [*]addrspace(.global) const u32,
    image: [*]addrspace(.global) const u32,
    result: [*]addrspace(.global) Uint2,
    block_offset: i32,
) callconv(.kernel) void {
    const idx = @workItemId(0);

    loadColorBlock(image, block_offset);

    std.nvptx.barrierSyncAligned(.work_group, null);

    var best_start: u16 = 0;
    var best_end: u16 = 0;
    var best_permutation: u32 = 0;

    evalAllPermutations(permutations, &best_start, &best_end, &best_permutation, sums[0]);

    // Use a parallel reduction to find the minimum error.
    const min_idx = findMinError();

    std.nvptx.barrierSyncAligned(.work_group, null);

    // Only the winner work item writes the result.
    if (idx == min_idx) {
        saveBlockDxt1(best_start, best_end, best_permutation, result, block_offset);
    }
}
