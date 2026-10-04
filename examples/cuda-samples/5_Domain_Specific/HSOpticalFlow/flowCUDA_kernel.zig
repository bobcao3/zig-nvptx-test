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

//! Variational optical flow estimation example. Uses textures for image operations. Shows how
//! simple PDE solver can be accelerated with CUDA.

const std = @import("std");

inline fn workItem(comptime dimension: u32) i32 {
    return @intCast(@workItemId(dimension));
}

inline fn globalIndex(comptime dimension: u32) i32 {
    const group_size: i32 = @intCast(@workGroupSize(dimension));
    const group: i32 = @intCast(@workGroupId(dimension));
    return workItem(dimension) + group * group_size;
}

inline fn tex2D(texture: std.nvptx.TextureObject, x: f32, y: f32) f32 {
    return texture.tex(.@"2d", f32, f32, .{ x, y })[0];
}

/// Adds two vectors of size `count`.
export fn addKernel(
    op1: [*]addrspace(.global) const f32,
    op2: [*]addrspace(.global) const f32,
    count: i32,
    sum: [*]addrspace(.global) f32,
) callconv(.kernel) void {
    const pos = globalIndex(0);

    if (pos >= count) return;

    const index: usize = @intCast(pos);
    sum[index] = op1[index] + op2[index];
}

/// Computes image derivatives. Relies heavily on the texture unit.
export fn computeDerivativesKernel(
    width: i32,
    height: i32,
    stride: i32,
    ix_out: [*]addrspace(.global) f32,
    iy_out: [*]addrspace(.global) f32,
    iz_out: [*]addrspace(.global) f32,
    tex_source: std.nvptx.TextureObject,
    tex_target: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const ix = globalIndex(0);
    const iy = globalIndex(1);

    const pos: usize = @intCast(ix + iy * stride);

    if (ix >= width or iy >= height) return;

    const dx = 1.0 / @as(f32, @floatFromInt(width));
    const dy = 1.0 / @as(f32, @floatFromInt(height));

    const x = (@as(f32, @floatFromInt(ix)) + 0.5) * dx;
    const y = (@as(f32, @floatFromInt(iy)) + 0.5) * dy;

    var t0: f32 = undefined;
    var t1: f32 = undefined;
    t0 = tex2D(tex_source, x - 2.0 * dx, y);
    t0 -= tex2D(tex_source, x - 1.0 * dx, y) * 8.0;
    t0 += tex2D(tex_source, x + 1.0 * dx, y) * 8.0;
    t0 -= tex2D(tex_source, x + 2.0 * dx, y);
    t0 /= 12.0;

    t1 = tex2D(tex_target, x - 2.0 * dx, y);
    t1 -= tex2D(tex_target, x - 1.0 * dx, y) * 8.0;
    t1 += tex2D(tex_target, x + 1.0 * dx, y) * 8.0;
    t1 -= tex2D(tex_target, x + 2.0 * dx, y);
    t1 /= 12.0;

    ix_out[pos] = (t0 + t1) * 0.5;

    iz_out[pos] = tex2D(tex_target, x, y) - tex2D(tex_source, x, y);

    t0 = tex2D(tex_source, x, y - 2.0 * dy);
    t0 -= tex2D(tex_source, x, y - 1.0 * dy) * 8.0;
    t0 += tex2D(tex_source, x, y + 1.0 * dy) * 8.0;
    t0 -= tex2D(tex_source, x, y + 2.0 * dy);
    t0 /= 12.0;

    t1 = tex2D(tex_target, x, y - 2.0 * dy);
    t1 -= tex2D(tex_target, x, y - 1.0 * dy) * 8.0;
    t1 += tex2D(tex_target, x, y + 1.0 * dy) * 8.0;
    t1 -= tex2D(tex_target, x, y + 2.0 * dy);
    t1 /= 12.0;

    iy_out[pos] = (t0 + t1) * 0.5;
}

/// Downscales an image. Relies heavily on the texture unit.
export fn downscaleKernel(
    width: i32,
    height: i32,
    stride: i32,
    out: [*]addrspace(.global) f32,
    tex_fine: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const ix = globalIndex(0);
    const iy = globalIndex(1);

    if (ix >= width or iy >= height) return;

    const dx = 1.0 / @as(f32, @floatFromInt(width));
    const dy = 1.0 / @as(f32, @floatFromInt(height));

    const x = (@as(f32, @floatFromInt(ix)) + 0.5) * dx;
    const y = (@as(f32, @floatFromInt(iy)) + 0.5) * dy;

    out[@intCast(ix + iy * stride)] = 0.25 *
        (tex2D(tex_fine, x - dx * 0.25, y) + tex2D(tex_fine, x + dx * 0.25, y) +
            tex2D(tex_fine, x, y - dy * 0.25) + tex2D(tex_fine, x, y + dy * 0.25));
}

/// One iteration of classical Horn-Schunck method. It is one iteration of Jacobi method for a
/// corresponding linear system. The comptime parameters describe the work group size.
fn JacobiIteration(comptime bx: i32, comptime by: i32) type {
    return struct {
        var du: [(bx + 2) * (by + 2)]f32 addrspace(.shared) = undefined;
        var dv: [(bx + 2) * (by + 2)]f32 addrspace(.shared) = undefined;

        fn jacobiIteration(
            du0: [*]addrspace(.global) const f32,
            dv0: [*]addrspace(.global) const f32,
            ix_in: [*]addrspace(.global) const f32,
            iy_in: [*]addrspace(.global) const f32,
            iz_in: [*]addrspace(.global) const f32,
            w: i32,
            h: i32,
            s: i32,
            alpha: f32,
            du1: [*]addrspace(.global) f32,
            dv1: [*]addrspace(.global) f32,
        ) callconv(.kernel) void {
            const tx = workItem(0);
            const ty = workItem(1);
            const ix = globalIndex(0);
            const iy = globalIndex(1);

            const pos: usize = @intCast(@min(ix, w - 1) + @min(iy, h - 1) * s);

            const sh_mem_pos: usize = @intCast(tx + 1 + (ty + 1) * (bx + 2));

            du[sh_mem_pos] = du0[pos];
            dv[sh_mem_pos] = dv0[pos];

            if (ty == 0) {
                const bsx = workGroup(0) * bx;
                const bsy = workGroup(1) * by;

                const x = @min(bsx + tx, w - 1);
                var y: i32 = @max(bsy - 1, 0);
                var gm_pos: usize = @intCast(y * s + x);
                var sm_pos: usize = @intCast(tx + 1);
                du[sm_pos] = du0[gm_pos];
                dv[sm_pos] = dv0[gm_pos];

                y = @min(bsy + by, h - 1);
                sm_pos += (by + 1) * (bx + 2);
                gm_pos = @intCast(y * s + x);
                du[sm_pos] = du0[gm_pos];
                dv[sm_pos] = dv0[gm_pos];
            } else if (ty == 1) {
                const bsx = workGroup(0) * bx;
                const bsy = workGroup(1) * by;

                const y = @min(bsy + tx, h - 1);
                var x: i32 = @max(bsx - 1, 0);
                var sm_pos: usize = @intCast(bx + 2 + tx * (bx + 2));
                var gm_pos: usize = @intCast(x + y * s);

                if (tx < by) {
                    du[sm_pos] = du0[gm_pos];
                    dv[sm_pos] = dv0[gm_pos];
                    x = @min(bsx + bx, w - 1);
                    gm_pos = @intCast(y * s + x);
                    sm_pos += bx + 1;
                    du[sm_pos] = du0[gm_pos];
                    dv[sm_pos] = dv0[gm_pos];
                }
            }

            std.nvptx.barrierSyncAligned(.work_group, null);

            if (ix >= w or iy >= h) return;

            const left = sh_mem_pos - 1;
            const right = sh_mem_pos + 1;
            const up = sh_mem_pos + bx + 2;
            const down = sh_mem_pos - bx - 2;

            const sum_u = (du[left] + du[right] + du[up] + du[down]) * 0.25;
            const sum_v = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25;

            const frac = (ix_in[pos] * sum_u + iy_in[pos] * sum_v + iz_in[pos]) /
                (ix_in[pos] * ix_in[pos] + iy_in[pos] * iy_in[pos] + alpha);

            du1[pos] = sum_u - ix_in[pos] * frac;
            dv1[pos] = sum_v - iy_in[pos] * frac;
        }

        inline fn workGroup(comptime dimension: u32) i32 {
            return @intCast(@workGroupId(dimension));
        }

        comptime {
            @export(&jacobiIteration, .{ .name = "jacobiIteration" });
        }
    };
}

comptime {
    _ = JacobiIteration(32, 6);
}

/// Upscales one component of a displacement field.
export fn upscaleKernel(
    width: i32,
    height: i32,
    stride: i32,
    scale: f32,
    out: [*]addrspace(.global) f32,
    tex_coarse: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const ix = globalIndex(0);
    const iy = globalIndex(1);

    if (ix >= width or iy >= height) return;

    const x = (@as(f32, @floatFromInt(ix)) + 0.5) / @as(f32, @floatFromInt(width));
    const y = (@as(f32, @floatFromInt(iy)) + 0.5) / @as(f32, @floatFromInt(height));

    out[@intCast(ix + iy * stride)] = tex2D(tex_coarse, x, y) * scale;
}

/// Warps an image with a given displacement field.
export fn warpingKernel(
    width: i32,
    height: i32,
    stride: i32,
    u: [*]addrspace(.global) const f32,
    v: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    tex_to_warp: std.nvptx.TextureObject,
) callconv(.kernel) void {
    const ix = globalIndex(0);
    const iy = globalIndex(1);

    const pos: usize = @intCast(ix + iy * stride);

    if (ix >= width or iy >= height) return;

    const x = (@as(f32, @floatFromInt(ix)) + u[pos] + 0.5) / @as(f32, @floatFromInt(width));
    const y = (@as(f32, @floatFromInt(iy)) + v[pos] + 0.5) / @as(f32, @floatFromInt(height));

    out[pos] = tex2D(tex_to_warp, x, y);
}
