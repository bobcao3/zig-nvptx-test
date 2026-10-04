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

//! Device implementation of the Horn-Schunck optical flow method. Handles memory allocations and
//! control flow.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("common.zig");

const Kernels = struct {
    add: cuda.Function,
    compute_derivatives: cuda.Function,
    downscale: cuda.Function,
    jacobi_iteration: cuda.Function,
    upscale: cuda.Function,
    warping: cuda.Function,
    textures: std.ArrayList(cuda.TextureObject) = .empty,

    fn load(module: cuda.Module) !Kernels {
        return .{
            .add = try module.function("addKernel"),
            .compute_derivatives = try module.function("computeDerivativesKernel"),
            .downscale = try module.function("downscaleKernel"),
            .jacobi_iteration = try module.function("jacobiIteration"),
            .upscale = try module.function("upscaleKernel"),
            .warping = try module.function("warpingKernel"),
        };
    }

    fn deinit(kernels: *Kernels, gpa: std.mem.Allocator) void {
        for (kernels.textures.items) |texture| texture.destroy();
        kernels.textures.deinit(gpa);
    }

    /// Creates a texture that samples the image `src` with bilinear interpolation and mirrored
    /// out-of-range coordinates. The texture lives until `deinit`, because the launches that use
    /// it run asynchronously.
    fn mirroredTexture(
        kernels: *Kernels,
        gpa: std.mem.Allocator,
        src: cuda.Buffer(f32),
        shape: Shape,
    ) !cuda.TextureObject {
        const texture = try cuda.TextureObject.create(.{ .pitch2d = .{
            .address = src.address,
            .format = .float,
            .channels = 1,
            .width = shape.w,
            .height = shape.h,
            .pitch_bytes = shape.s * @sizeOf(f32),
        } }, .{
            .normalized_coordinates = true,
            .filter_mode = .linear,
            .address_mode = .{ .mirror, .mirror, .clamp },
            .read_mode = .element_type,
        });
        errdefer texture.destroy();
        try kernels.textures.append(gpa, texture);
        return texture;
    }
};

const Shape = struct {
    w: u32,
    h: u32,
    s: u32,

    fn blocks(shape: Shape, threads: cuda.Dim3) cuda.Dim3 {
        return .{ .x = common.iDivUp(shape.w, threads.x), .y = common.iDivUp(shape.h, threads.y) };
    }

    fn args(shape: Shape) struct { i32, i32, i32 } {
        return .{ @intCast(shape.w), @intCast(shape.h), @intCast(shape.s) };
    }
};

/// Adds two vectors of size `count`.
fn add(
    kernels: Kernels,
    op1: cuda.Buffer(f32),
    op2: cuda.Buffer(f32),
    count: u32,
    sum: cuda.Buffer(f32),
) !void {
    const threads: cuda.Dim3 = .{ .x = 256 };
    try kernels.add.launch(.{
        .grid = .{ .x = common.iDivUp(count, threads.x) },
        .block = threads,
    }, .{ op1, op2, @as(i32, @intCast(count)), sum });
}

fn computeDerivatives(
    kernels: *Kernels,
    gpa: std.mem.Allocator,
    image0: cuda.Buffer(f32),
    image1: cuda.Buffer(f32),
    shape: Shape,
    ix: cuda.Buffer(f32),
    iy: cuda.Buffer(f32),
    iz: cuda.Buffer(f32),
) !void {
    const threads: cuda.Dim3 = .{ .x = 32, .y = 6 };

    const tex_source = try kernels.mirroredTexture(gpa, image0, shape);
    const tex_target = try kernels.mirroredTexture(gpa, image1, shape);

    try kernels.compute_derivatives.launch(
        .{ .grid = shape.blocks(threads), .block = threads },
        shape.args() ++ .{ ix, iy, iz, tex_source, tex_target },
    );
}

fn downscale(
    kernels: *Kernels,
    gpa: std.mem.Allocator,
    src: cuda.Buffer(f32),
    shape: Shape,
    new_shape: Shape,
    out: cuda.Buffer(f32),
) !void {
    const threads: cuda.Dim3 = .{ .x = 32, .y = 8 };

    const tex_fine = try kernels.mirroredTexture(gpa, src, shape);

    try kernels.downscale.launch(
        .{ .grid = new_shape.blocks(threads), .block = threads },
        new_shape.args() ++ .{ out, tex_fine },
    );
}

/// One iteration of classical Horn-Schunck method. It is one iteration of Jacobi method for a
/// corresponding linear system.
fn solveForUpdate(
    kernels: Kernels,
    du0: cuda.Buffer(f32),
    dv0: cuda.Buffer(f32),
    ix: cuda.Buffer(f32),
    iy: cuda.Buffer(f32),
    iz: cuda.Buffer(f32),
    shape: Shape,
    alpha: f32,
    du1: cuda.Buffer(f32),
    dv1: cuda.Buffer(f32),
) !void {
    const threads: cuda.Dim3 = .{ .x = 32, .y = 6 };

    try kernels.jacobi_iteration.launch(
        .{ .grid = shape.blocks(threads), .block = threads },
        .{ du0, dv0, ix, iy, iz } ++ shape.args() ++ .{ alpha, du1, dv1 },
    );
}

/// Upscales one component of a displacement field.
fn upscale(
    kernels: *Kernels,
    gpa: std.mem.Allocator,
    src: cuda.Buffer(f32),
    shape: Shape,
    new_shape: Shape,
    scale: f32,
    out: cuda.Buffer(f32),
) !void {
    const threads: cuda.Dim3 = .{ .x = 32, .y = 8 };

    const tex_coarse = try kernels.mirroredTexture(gpa, src, shape);

    try kernels.upscale.launch(
        .{ .grid = new_shape.blocks(threads), .block = threads },
        new_shape.args() ++ .{ scale, out, tex_coarse },
    );
}

/// Warps an image with a provided vector field. For each output pixel there is a vector which
/// tells which pixel from a source image should be mapped to this particular output pixel. It is
/// assumed that images and the vector field have the same stride and resolution.
fn warpImage(
    kernels: *Kernels,
    gpa: std.mem.Allocator,
    src: cuda.Buffer(f32),
    shape: Shape,
    u: cuda.Buffer(f32),
    v: cuda.Buffer(f32),
    out: cuda.Buffer(f32),
) !void {
    const threads: cuda.Dim3 = .{ .x = 32, .y = 6 };

    const tex_to_warp = try kernels.mirroredTexture(gpa, src, shape);

    try kernels.warping.launch(
        .{ .grid = shape.blocks(threads), .block = threads },
        shape.args() ++ .{ u, v, out, tex_to_warp },
    );
}

fn ratio(numerator: u32, denominator: u32) f32 {
    return @as(f32, @floatFromInt(numerator)) / @as(f32, @floatFromInt(denominator));
}

const Level = struct {
    image0: cuda.Buffer(f32),
    image1: cuda.Buffer(f32),
    shape: Shape,
};

/// Method logic. Handles memory allocations and control flow.
pub fn computeFlowCuda(
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
    std.log.info("Computing optical flow on GPU...", .{});

    const module = try cuda.Module.load(@embedFile("flowCUDA_kernel.ptx"));
    defer module.unload();
    var kernels: Kernels = try .load(module);
    defer kernels.deinit(gpa);

    const pyramid = try gpa.alloc(Level, n_levels);
    defer gpa.free(pyramid);
    var created_levels: usize = 0;
    defer for (pyramid[0..created_levels]) |level| {
        level.image0.free();
        level.image1.free();
    };

    const data_size = stride * height;

    const d_tmp = try cuda.alloc(f32, data_size);
    defer d_tmp.free();
    var d_du0 = try cuda.alloc(f32, data_size);
    defer d_du0.free();
    var d_dv0 = try cuda.alloc(f32, data_size);
    defer d_dv0.free();
    var d_du1 = try cuda.alloc(f32, data_size);
    defer d_du1.free();
    var d_dv1 = try cuda.alloc(f32, data_size);
    defer d_dv1.free();

    const d_ix = try cuda.alloc(f32, data_size);
    defer d_ix.free();
    const d_iy = try cuda.alloc(f32, data_size);
    defer d_iy.free();
    const d_iz = try cuda.alloc(f32, data_size);
    defer d_iz.free();

    var d_u = try cuda.alloc(f32, data_size);
    defer d_u.free();
    var d_v = try cuda.alloc(f32, data_size);
    defer d_v.free();
    var d_nu = try cuda.alloc(f32, data_size);
    defer d_nu.free();
    var d_nv = try cuda.alloc(f32, data_size);
    defer d_nv.free();

    var current_level = n_levels - 1;
    pyramid[current_level] = top: {
        const top_image0 = try cuda.dupe(f32, image0);
        errdefer top_image0.free();
        break :top .{
            .image0 = top_image0,
            .image1 = try cuda.dupe(f32, image1),
            .shape = .{ .w = width, .h = height, .s = stride },
        };
    };
    created_levels += 1;

    while (current_level > 0) : (current_level -= 1) {
        const fine = pyramid[current_level];
        const nw = fine.shape.w / 2;
        const nh = fine.shape.h / 2;
        const new_shape: Shape = .{ .w = nw, .h = nh, .s = common.iAlignUp(nw) };

        const coarse = &pyramid[current_level - 1];
        coarse.* = next: {
            const next_image0 = try cuda.alloc(f32, new_shape.s * nh);
            errdefer next_image0.free();
            break :next .{
                .image0 = next_image0,
                .image1 = try cuda.alloc(f32, new_shape.s * nh),
                .shape = new_shape,
            };
        };
        created_levels += 1;

        try downscale(&kernels, gpa, fine.image0, fine.shape, new_shape, coarse.image0);
        try downscale(&kernels, gpa, fine.image1, fine.shape, new_shape, coarse.image1);
    }

    try d_u.memset(0);
    try d_v.memset(0);

    while (current_level < n_levels) : (current_level += 1) {
        const level = pyramid[current_level];
        const shape = level.shape;
        for (0..n_warp_iters) |_| {
            try d_du0.memset(0);
            try d_dv0.memset(0);

            try d_du1.memset(0);
            try d_dv1.memset(0);

            try warpImage(&kernels, gpa, level.image1, level.shape, d_u, d_v, d_tmp);

            try computeDerivatives(&kernels, gpa, level.image0, d_tmp, shape, d_ix, d_iy, d_iz);

            for (0..n_solver_iters) |_| {
                try solveForUpdate(
                    kernels,
                    d_du0,
                    d_dv0,
                    d_ix,
                    d_iy,
                    d_iz,
                    shape,
                    alpha,
                    d_du1,
                    d_dv1,
                );

                std.mem.swap(cuda.Buffer(f32), &d_du0, &d_du1);
                std.mem.swap(cuda.Buffer(f32), &d_dv0, &d_dv1);
            }

            try add(kernels, d_u, d_du0, shape.h * shape.s, d_u);
            try add(kernels, d_v, d_dv0, shape.h * shape.s, d_v);
        }

        if (current_level != n_levels - 1) {
            const next = pyramid[current_level + 1];
            const scale_x = ratio(next.shape.w, shape.w);

            try upscale(&kernels, gpa, d_u, shape, next.shape, scale_x, d_nu);

            const scale_y = ratio(next.shape.h, shape.h);

            try upscale(&kernels, gpa, d_v, shape, next.shape, scale_y, d_nv);

            std.mem.swap(cuda.Buffer(f32), &d_u, &d_nu);
            std.mem.swap(cuda.Buffer(f32), &d_v, &d_nv);
        }
    }

    try d_u.copyTo(u_out);
    try d_v.copyTo(v_out);
}
