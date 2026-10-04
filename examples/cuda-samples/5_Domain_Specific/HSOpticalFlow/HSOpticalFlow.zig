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
const cuda = @import("cuda");
const common = @import("common.zig");
const flow_cuda = @import("flowCUDA.zig");
const flow_gold = @import("flowGold.zig");

const sample_name = "HSOpticalFlow";

/// CPU-GPU discrepancy threshold for self-test.
const threshold = 0.05;

/// Saves optical flow in format described on vision.middlebury.edu/flow.
fn writeFloFile(
    gpa: std.mem.Allocator,
    io: std.Io,
    name: []const u8,
    w: u32,
    h: u32,
    s: u32,
    u: []const f32,
    v: []const f32,
) !void {
    var contents: std.Io.Writer.Allocating = .init(gpa);
    defer contents.deinit();
    const writer = &contents.writer;

    try writer.writeInt(u32, @bitCast(@as(f32, 202021.25)), .little);
    try writer.writeInt(u32, w, .little);
    try writer.writeInt(u32, h, .little);

    for (0..h) |i| {
        for (0..w) |j| {
            const pos = j + i * s;
            try writer.writeInt(u32, @bitCast(u[pos]), .little);
            try writer.writeInt(u32, @bitCast(v[pos]), .little);
        }
    }

    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = contents.written() }) catch {
        std.log.warn("Could not save flow to \"{s}\"", .{name});
    };
}

const Image = struct {
    data: []f32,
    width: u32,
    height: u32,
    stride: u32,
};

/// Loads a 4-channel unsigned byte image and converts it to a single channel FP32 image.
fn loadImageAsFp32(gpa: std.mem.Allocator, name: []const u8, file: []const u8) !Image {
    std.log.info("Loading \"{s}\" ...", .{name});

    var tokens = std.mem.tokenizeAny(u8, file, " \t\r\n");
    if (!std.mem.eql(u8, tokens.next() orelse return error.InvalidImage, "P6")) {
        std.log.err("Invalid file format", .{});
        return error.InvalidImage;
    }
    var fields: [3]u32 = undefined;
    for (&fields) |*field| {
        var token = tokens.next() orelse return error.InvalidImage;
        while (token[0] == '#') {
            tokens.index = std.mem.indexOfScalarPos(u8, file, tokens.index, '\n') orelse
                return error.InvalidImage;
            token = tokens.next() orelse return error.InvalidImage;
        }
        field.* = try std.fmt.parseInt(u32, token, 10);
    }
    const w, const h, _ = fields;
    const pixels = file[tokens.index + 1 ..][0 .. 3 * w * h];

    const stride = common.iAlignUp(w);
    const data = try gpa.alloc(f32, stride * h);

    const width_step = 3 * w;

    for (0..h) |i| {
        for (0..w) |j| {
            data[j + i * stride] = @as(f32, @floatFromInt(pixels[j * 3 + i * width_step])) / 255.0;
        }
    }

    return .{ .data = data, .width = w, .height = h, .stride = stride };
}

/// Compares the given flow field with gold (L1 norm). Returns whether the discrepancy is lower
/// than the threshold.
fn compareWithGold(
    width: u32,
    height: u32,
    stride: u32,
    h_u_gold: []const f32,
    h_v_gold: []const f32,
    h_u: []const f32,
    h_v: []const f32,
) bool {
    var err: f32 = 0.0;

    for (0..height) |i| {
        for (0..width) |j| {
            const pos = j + i * stride;
            err += @abs(h_u[pos] - h_u_gold[pos]) + @abs(h_v[pos] - h_v_gold[pos]);
        }
    }

    err /= @floatFromInt(width * height);

    std.log.info("L1 error : {d:.6}", .{err});

    return err < threshold;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("{s} Starting...", .{sample_name});

    const device = try cuda.init(0);
    defer device.deinit();

    const source_frame_name = "frame10.ppm";
    const target_frame_name = "frame11.ppm";

    const source = loadImageAsFp32(gpa, source_frame_name, @embedFile(source_frame_name)) catch
        std.process.exit(1);
    defer gpa.free(source.data);
    const target = loadImageAsFp32(gpa, target_frame_name, @embedFile(target_frame_name)) catch
        std.process.exit(1);
    defer gpa.free(target.data);
    const width = source.width;
    const height = source.height;
    const stride = source.stride;

    const h_u_gold = try gpa.alloc(f32, stride * height);
    defer gpa.free(h_u_gold);
    const h_v_gold = try gpa.alloc(f32, stride * height);
    defer gpa.free(h_v_gold);

    const h_u = try gpa.alloc(f32, stride * height);
    defer gpa.free(h_u);
    const h_v = try gpa.alloc(f32, stride * height);
    defer gpa.free(h_v);

    const alpha = 0.2;

    const n_levels = 5;

    const n_solver_iters = 500;

    const n_warp_iters = 3;

    try flow_gold.computeFlowGold(
        gpa,
        source.data,
        target.data,
        width,
        height,
        stride,
        alpha,
        n_levels,
        n_warp_iters,
        n_solver_iters,
        h_u_gold,
        h_v_gold,
    );

    try flow_cuda.computeFlowCuda(
        gpa,
        source.data,
        target.data,
        width,
        height,
        stride,
        alpha,
        n_levels,
        n_warp_iters,
        n_solver_iters,
        h_u,
        h_v,
    );

    const status = compareWithGold(width, height, stride, h_u_gold, h_v_gold, h_u, h_v);

    try writeFloFile(gpa, init.io, "FlowGPU.flo", width, height, stride, h_u, h_v);

    try writeFloFile(gpa, init.io, "FlowCPU.flo", width, height, stride, h_u_gold, h_v_gold);

    if (!status) std.process.exit(1);
}
