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

//! Host implementation of the Horn-Schunck optical flow method, used as the reference for the
//! device implementation.

const std = @import("std");
const common = @import("common.zig");

/// Reads from arbitrary position within image using bilinear interpolation. Out of range
/// coordinates are mirrored.
inline fn tex2D(t: []const f32, w: i32, h: i32, s: i32, x: f32, y: f32) f32 {
    const x_parts = std.math.modf(x);
    const y_parts = std.math.modf(y);
    const dx = @abs(x_parts.fpart);
    const dy = @abs(y_parts.fpart);

    var ix0: i32 = @intFromFloat(x_parts.ipart);
    var iy0: i32 = @intFromFloat(y_parts.ipart);

    if (ix0 < 0) ix0 = @intCast(@abs(ix0 + 1));

    if (iy0 < 0) iy0 = @intCast(@abs(iy0 + 1));

    if (ix0 >= w) ix0 = w * 2 - ix0 - 1;

    if (iy0 >= h) iy0 = h * 2 - iy0 - 1;

    var ix1 = ix0 + 1;
    var iy1 = iy0 + 1;

    if (ix1 >= w) ix1 = w * 2 - ix1 - 1;

    if (iy1 >= h) iy1 = h * 2 - iy1 - 1;

    var res = t[index(ix0 + iy0 * s)] * (1.0 - dx) * (1.0 - dy);
    res += t[index(ix1 + iy0 * s)] * dx * (1.0 - dy);
    res += t[index(ix0 + iy1 * s)] * (1.0 - dx) * dy;
    res += t[index(ix1 + iy1 * s)] * dx * dy;

    return res;
}

/// Reads a specific texel value. Out of range coordinates are mirrored.
inline fn tex2Di(src: []const f32, w: i32, h: i32, s: i32, x_in: i32, y_in: i32) f32 {
    var x = x_in;
    var y = y_in;
    if (x < 0) x = @intCast(@abs(x + 1));

    if (y < 0) y = @intCast(@abs(y + 1));

    if (x >= w) x = w * 2 - x - 1;

    if (y >= h) y = h * 2 - y - 1;

    return src[index(x + y * s)];
}

fn index(position: i32) usize {
    return @intCast(position);
}

fn downscale(
    src: []const f32,
    width: i32,
    height: i32,
    stride: i32,
    new_width: i32,
    new_height: i32,
    new_stride: i32,
    out: []f32,
) void {
    var i: i32 = 0;
    while (i < new_height) : (i += 1) {
        var j: i32 = 0;
        while (j < new_width) : (j += 1) {
            const src_x = j * 2;
            const src_y = i * 2;
            var sum: f32 = tex2Di(src, width, height, stride, src_x + 0, src_y + 0);
            sum += tex2Di(src, width, height, stride, src_x + 0, src_y + 1);
            sum += tex2Di(src, width, height, stride, src_x + 1, src_y + 0);
            sum += tex2Di(src, width, height, stride, src_x + 1, src_y + 1);
            sum *= 0.25;
            out[index(j + i * new_stride)] = sum;
        }
    }
}

fn upscale(
    src: []const f32,
    width: i32,
    height: i32,
    stride: i32,
    new_width: i32,
    new_height: i32,
    new_stride: i32,
    scale: f32,
    out: []f32,
) void {
    var i: i32 = 0;
    while (i < new_height) : (i += 1) {
        var j: i32 = 0;
        while (j < new_width) : (j += 1) {
            const x = (@as(f32, @floatFromInt(j)) - 0.5) * 0.5;
            const y = (@as(f32, @floatFromInt(i)) - 0.5) * 0.5;

            out[index(j + i * new_stride)] = tex2D(src, width, height, stride, x, y) * scale;
        }
    }
}

/// Warps an image with a provided vector field. For each output pixel there is a vector which
/// tells which pixel from a source image should be mapped to this particular output pixel. It is
/// assumed that images and the vector field have the same stride and resolution.
fn warpImage(
    src: []const f32,
    w: i32,
    h: i32,
    s: i32,
    u: []const f32,
    v: []const f32,
    out: []f32,
) void {
    var i: i32 = 0;
    while (i < h) : (i += 1) {
        var j: i32 = 0;
        while (j < w) : (j += 1) {
            const pos = index(j + i * s);
            const x = @as(f32, @floatFromInt(j)) + u[pos];
            const y = @as(f32, @floatFromInt(i)) + v[pos];

            out[pos] = tex2D(src, w, h, s, x, y);
        }
    }
}

fn computeDerivatives(
    image0: []const f32,
    image1: []const f32,
    w: i32,
    h: i32,
    s: i32,
    ix: []f32,
    iy: []f32,
    iz: []f32,
) void {
    var i: i32 = 0;
    while (i < h) : (i += 1) {
        var j: i32 = 0;
        while (j < w) : (j += 1) {
            const pos = index(j + i * s);
            var t0: f32 = undefined;
            var t1: f32 = undefined;
            t0 = tex2Di(image0, w, h, s, j - 2, i);
            t0 -= tex2Di(image0, w, h, s, j - 1, i) * 8.0;
            t0 += tex2Di(image0, w, h, s, j + 1, i) * 8.0;
            t0 -= tex2Di(image0, w, h, s, j + 2, i);
            t0 /= 12.0;

            t1 = tex2Di(image1, w, h, s, j - 2, i);
            t1 -= tex2Di(image1, w, h, s, j - 1, i) * 8.0;
            t1 += tex2Di(image1, w, h, s, j + 1, i) * 8.0;
            t1 -= tex2Di(image1, w, h, s, j + 2, i);
            t1 /= 12.0;

            ix[pos] = (t0 + t1) * 0.5;

            iz[pos] = image1[pos] - image0[pos];

            t0 = tex2Di(image0, w, h, s, j, i - 2);
            t0 -= tex2Di(image0, w, h, s, j, i - 1) * 8.0;
            t0 += tex2Di(image0, w, h, s, j, i + 1) * 8.0;
            t0 -= tex2Di(image0, w, h, s, j, i + 2);
            t0 /= 12.0;

            t1 = tex2Di(image1, w, h, s, j, i - 2);
            t1 -= tex2Di(image1, w, h, s, j, i - 1) * 8.0;
            t1 += tex2Di(image1, w, h, s, j, i + 1) * 8.0;
            t1 -= tex2Di(image1, w, h, s, j, i + 2);
            t1 /= 12.0;

            iy[pos] = (t0 + t1) * 0.5;
        }
    }
}

/// One iteration of classical Horn-Schunck method. It is one iteration of Jacobi method for a
/// corresponding linear system.
fn solveForUpdate(
    du0: []const f32,
    dv0: []const f32,
    ix: []const f32,
    iy: []const f32,
    iz: []const f32,
    w: i32,
    h: i32,
    s: i32,
    alpha: f32,
    du1: []f32,
    dv1: []f32,
) void {
    var i: i32 = 0;
    while (i < h) : (i += 1) {
        var j: i32 = 0;
        while (j < w) : (j += 1) {
            const pos = j + i * s;

            const left = if (j != 0) pos - 1 else pos;
            const right = if (j != w - 1) pos + 1 else pos;
            const down = if (i != 0) pos - s else pos;
            const up = if (i != h - 1) pos + s else pos;

            const sum_u = (du0[index(left)] + du0[index(right)] +
                du0[index(up)] + du0[index(down)]) * 0.25;
            const sum_v = (dv0[index(left)] + dv0[index(right)] +
                dv0[index(up)] + dv0[index(down)]) * 0.25;

            const p = index(pos);
            const frac = (ix[p] * sum_u + iy[p] * sum_v + iz[p]) /
                (ix[p] * ix[p] + iy[p] * iy[p] + alpha);

            du1[p] = sum_u - ix[p] * frac;
            dv1[p] = sum_v - iy[p] * frac;
        }
    }
}

const Level = struct {
    image0: []const f32,
    image1: []const f32,
    w: i32,
    h: i32,
    s: i32,
};

/// Method logic. Handles memory allocation and control flow.
pub fn computeFlowGold(
    gpa: std.mem.Allocator,
    image0: []const f32,
    image1: []const f32,
    width: u32,
    height: u32,
    stride: u32,
    alpha: f32,
    n_levels: u32,
    n_warp_iters: u32,
    n_solver_iters: u32,
    u_out: []f32,
    v_out: []f32,
) !void {
    std.log.info("Computing optical flow on CPU...", .{});

    var u = u_out;
    var v = v_out;

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const pyramid = arena.allocator();

    const levels = try pyramid.alloc(Level, n_levels);

    const pixel_count_aligned = height * stride;

    const tmp = try pyramid.alloc(f32, pixel_count_aligned);
    var du0 = try pyramid.alloc(f32, pixel_count_aligned);
    var dv0 = try pyramid.alloc(f32, pixel_count_aligned);
    var du1 = try pyramid.alloc(f32, pixel_count_aligned);
    var dv1 = try pyramid.alloc(f32, pixel_count_aligned);
    const ix = try pyramid.alloc(f32, pixel_count_aligned);
    const iy = try pyramid.alloc(f32, pixel_count_aligned);
    const iz = try pyramid.alloc(f32, pixel_count_aligned);
    var nu = try pyramid.alloc(f32, pixel_count_aligned);
    var nv = try pyramid.alloc(f32, pixel_count_aligned);

    var current_level = n_levels - 1;
    levels[current_level] = .{
        .image0 = image0,
        .image1 = image1,
        .w = @intCast(width),
        .h = @intCast(height),
        .s = @intCast(stride),
    };

    while (current_level > 0) : (current_level -= 1) {
        const fine = levels[current_level];
        const nw = @divTrunc(fine.w, 2);
        const nh = @divTrunc(fine.h, 2);
        const ns: i32 = @intCast(common.iAlignUp(@intCast(nw)));
        const next_image0 = try pyramid.alloc(f32, @intCast(ns * nh));
        const next_image1 = try pyramid.alloc(f32, @intCast(ns * nh));

        downscale(fine.image0, fine.w, fine.h, fine.s, nw, nh, ns, next_image0);
        downscale(fine.image1, fine.w, fine.h, fine.s, nw, nh, ns, next_image1);

        levels[current_level - 1] = .{
            .image0 = next_image0,
            .image1 = next_image1,
            .w = nw,
            .h = nh,
            .s = ns,
        };
    }

    @memset(u, 0);
    @memset(v, 0);

    while (current_level < n_levels) : (current_level += 1) {
        const level = levels[current_level];
        const w = level.w;
        const h = level.h;
        const s = level.s;
        for (0..n_warp_iters) |_| {
            @memset(du0, 0);
            @memset(dv0, 0);

            @memset(du1, 0);
            @memset(dv1, 0);

            warpImage(level.image1, w, h, s, u, v, tmp);

            computeDerivatives(level.image0, tmp, w, h, s, ix, iy, iz);

            for (0..n_solver_iters) |_| {
                solveForUpdate(du0, dv0, ix, iy, iz, w, h, s, alpha, du1, dv1);
                std.mem.swap([]f32, &du0, &du1);
                std.mem.swap([]f32, &dv0, &dv1);
            }

            for (0..@intCast(h * s)) |i| {
                u[i] += du0[i];
                v[i] += dv0[i];
            }
        }

        if (current_level != n_levels - 1) {
            const next = levels[current_level + 1];
            const next_w = next.w;
            const next_h = next.h;
            const next_s = next.s;
            const scale_x = @as(f32, @floatFromInt(next_w)) / @as(f32, @floatFromInt(w));

            upscale(u, w, h, s, next_w, next_h, next_s, scale_x, nu);

            const scale_y = @as(f32, @floatFromInt(next_h)) / @as(f32, @floatFromInt(h));

            upscale(v, w, h, s, next_w, next_h, next_s, scale_y, nv);

            std.mem.swap([]f32, &u, &nu);
            std.mem.swap([]f32, &v, &nv);
        }
    }

    if (u.ptr != u_out.ptr) {
        @memcpy(u_out, u[0..pixel_count_aligned]);
        @memcpy(v_out, v[0..pixel_count_aligned]);
    }
}
