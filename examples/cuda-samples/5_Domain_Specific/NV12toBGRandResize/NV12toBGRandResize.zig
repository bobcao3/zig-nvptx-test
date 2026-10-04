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

//! NVIDIA HW Decoder, both dGPU and Tegra, normally outputs NV12 pitch format frames. For the
//! inference using TensorRT, the input frame needs to be BGR planar format with possibly
//! different size. So, conversion and resizing from NV12 to BGR planar is usually required for
//! the inference following decoding. This CUDA code is to provide a reference implementation for
//! conversion and resizing.
//!
//! Limitation: NV12resize needs the height to be a even value.
//!
//! Note: Resize function needs the pitch of image buffer to be 32 alignment.
//!
//! Run: `NV12toBGRandResize` or `NV12toBGRandResize -input=data/test1920x1080.nv12 -width=1920
//! -height=1080 -dst_width=640 -dst_height=480 -batch=40 -device=0`

const std = @import("std");
const cuda = @import("cuda");
const bgr_resize = @import("bgr_resize.zig");
const nv12_resize = @import("nv12_resize.zig");
const nv12_to_bgr_planar = @import("nv12_to_bgr_planar.zig");
const utils = @import("utils.zig");

const test_loop = 20;

const Context = struct {
    width: u32 = 0,
    height: u32 = 0,
    pitch: u32 = 0,

    dst_width: u32 = 0,
    dst_height: u32 = 0,
    dst_pitch: u32 = 0,

    batch: u32 = 0,
    device: u32 = 0,

    input_nv12_file: ?[]const u8 = null,

    /// The value will be suitable for Texture memory.
    ctx_pitch: u32 = 0,
    /// The value will be even.
    ctx_heights: u32 = 0,
};

const Kernels = struct {
    resize_nv12: cuda.Function,
    resize_bgr: cuda.Function,
    nv12_to_bgr: cuda.Function,
};

const average_format = ", batch: {d}, average time: {d:.3} ms ==> {d:.3} ms/frame";
const resize_nv12_format = "  CUDA resize nv12({d}x{d} --> {d}x{d})" ++ average_format;
const convert_nv12_format = "  CUDA convert nv12({d}x{d}) to bgr({d}x{d})" ++ average_format;
const resize_bgr_format = "  CUDA resize bgr({d}x{d} --> {d}x{d})" ++ average_format;

const default_input = "test1920x1080.nv12";

fn printHelp(app_name: []const u8) void {
    std.log.info(
        \\Usage:{s} [options]
        \\
        \\OPTIONS:
        \\    -h,--help
        \\
        \\    -input=nv12file             nv12 input file
        \\    -width=width                input nv12 image width, <1 -- 4096>
        \\    -height=height              input nv12 image height, <1 -- 4096>
        \\    -pitch=pitch(optional)      input nv12 image pitch, <0 -- 4096>
        \\    -dst_width=width            output BGR image width, <1 -- 4096>
        \\    -dst_height=height          output BGR image height, <1 -- 4096>
        \\    -dst_pitch=pitch(optional)  output BGR image pitch, <0 -- 4096>
        \\    -batch=batch                process frames count, <1 -- 4096>
        \\
        \\    -device=device_num(optional)   cuda device number, <0 -- 4096>
    , .{app_name});
}

/// Returns the value of the command line argument `-<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') return flag[name.len + 1 ..];
    }
    return null;
}

fn flagInt(args: []const [:0]const u8, name: []const u8) !?u32 {
    const value = flagValue(args, name) orelse return null;
    return try std.fmt.parseInt(u32, value, 10);
}

/// Returns null when the arguments do not describe a run.
fn parseCmdLine(args: []const [:0]const u8) !?Context {
    var ctx: Context = .{};

    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printHelp(args[0]);
            return null;
        }
    }

    if (args.len == 1) {
        ctx.input_nv12_file = default_input;
        ctx.width = 1920;
        ctx.height = 1080;
        ctx.dst_width = 640;
        ctx.dst_height = 480;
        ctx.batch = 24;
    } else {
        ctx.width = try flagInt(args, "width") orelse 0;
        ctx.height = try flagInt(args, "height") orelse 0;
        ctx.pitch = try flagInt(args, "pitch") orelse 0;
        ctx.input_nv12_file = flagValue(args, "input");
        ctx.dst_width = try flagInt(args, "dst_width") orelse 0;
        ctx.dst_height = try flagInt(args, "dst_height") orelse 0;
        ctx.dst_pitch = try flagInt(args, "dst_pitch") orelse 0;
        ctx.batch = try flagInt(args, "batch") orelse 0;
    }

    ctx.device = try flagInt(args, "device") orelse 0;

    if (ctx.width == 0 or ctx.height == 0 or ctx.dst_width == 0 or ctx.dst_height == 0 or
        ctx.input_nv12_file == null)
    {
        printHelp(args[0]);
        return null;
    }

    if (ctx.pitch == 0) ctx.pitch = ctx.width;
    if (ctx.dst_pitch == 0) ctx.dst_pitch = ctx.dst_width;

    return ctx;
}

/// Loads nv12 yuvfile data into GPU device memory with batch of copy.
fn loadNv12Frame(
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: Context,
    d_input_nv12: cuda.Buffer(u8),
) !void {
    const input_file = ctx.input_nv12_file.?;
    const file = if (std.mem.eql(u8, input_file, default_input))
        @as([]const u8, @embedFile(default_input))
    else
        std.Io.Dir.cwd().readFileAlloc(io, input_file, gpa, .unlimited) catch {
            std.log.err("Can't open files", .{});
            return error.CannotOpenInput;
        };
    defer if (file.ptr != @embedFile(default_input).ptr) gpa.free(file);

    const frame_size = ctx.pitch * ctx.ctx_heights;

    if (file.len < frame_size) {
        std.log.err("can't get one frame!", .{});
        return error.IncompleteFrame;
    }

    const frames = try gpa.alloc(u8, ctx.ctx_pitch * ctx.ctx_heights * ctx.batch);
    defer gpa.free(frames);
    @memset(frames, 0);
    for (0..ctx.batch) |i| {
        for (0..ctx.ctx_heights) |row| {
            const frame_row = (i * ctx.ctx_heights + row) * ctx.ctx_pitch;
            @memcpy(frames[frame_row..][0..ctx.width], file[row * ctx.width ..][0..ctx.width]);
        }
    }
    try d_input_nv12.copyFrom(frames);
}

fn printTiming(
    comptime format: []const u8,
    ctx: Context,
    sizes: [4]u32,
    elapsed_time: f32,
) void {
    const average = elapsed_time / (test_loop * 1.0);
    std.log.info(format, .{
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        ctx.batch,
        average,
        average / @as(f32, @floatFromInt(ctx.batch)),
    });
}

/// 1. resize interlace nv12 to target size
/// 2. convert nv12 to bgr 3 progressive planars
fn nv12ResizeAndNv12ToBgr(
    gpa: std.mem.Allocator,
    io: std.Io,
    kernels: Kernels,
    ctx: Context,
    d_input_nv12: cuda.Buffer(u8),
) !void {
    const dst_height_f: f32 = @floatFromInt(ctx.dst_height);
    const resized_height: u32 = @intFromFloat(@ceil(dst_height_f * 3.0 / 2.0));
    const d_resized_nv12 = try cuda.alloc(u8, ctx.dst_width * resized_height * ctx.batch);
    defer d_resized_nv12.free();

    const d_output_bgr = try cuda.alloc(f32, ctx.dst_pitch * ctx.dst_height * 3 * ctx.batch);
    defer d_output_bgr.free();

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    try start.record(.default);
    for (0..test_loop) |_| {
        try nv12_resize.resizeNv12Batch(
            kernels.resize_nv12,
            d_input_nv12,
            ctx.ctx_pitch,
            ctx.width,
            ctx.height,
            d_resized_nv12,
            ctx.dst_width,
            ctx.dst_width,
            ctx.dst_height,
            ctx.batch,
        );
    }
    try stop.record(.default);
    try stop.synchronize();

    var elapsed_time = try start.elapsed(stop);
    printTiming(
        resize_nv12_format,
        ctx,
        .{ ctx.width, ctx.height, ctx.dst_width, ctx.dst_height },
        elapsed_time,
    );

    try start.record(.default);
    for (0..test_loop) |_| {
        try nv12_to_bgr_planar.nv12ToBgrPlanarBatch(
            kernels.nv12_to_bgr,
            d_resized_nv12,
            ctx.dst_pitch,
            d_output_bgr,
            ctx.dst_pitch * @sizeOf(f32),
            ctx.dst_width,
            ctx.dst_height,
            ctx.batch,
        );
    }
    try stop.record(.default);
    try stop.synchronize();

    elapsed_time = try start.elapsed(stop);
    printTiming(
        convert_nv12_format,
        ctx,
        .{ ctx.dst_width, ctx.dst_height, ctx.dst_width, ctx.dst_height },
        elapsed_time,
    );

    var filename_buffer: [40]u8 = undefined;
    const filename = try std.fmt.bufPrint(&filename_buffer, "converted_bgr_{d}x{d}", .{
        ctx.dst_width,
        ctx.dst_height,
    });
    try utils.dumpBgr(
        gpa,
        io,
        d_output_bgr,
        ctx.dst_pitch,
        ctx.dst_width,
        ctx.dst_height,
        ctx.batch,
        "t1",
        filename,
    );
}

/// 1. convert nv12 to bgr 3 progressive planars
/// 2. resize bgr 3 planars to target size
fn nv12ToBgrAndBgrResize(
    gpa: std.mem.Allocator,
    io: std.Io,
    kernels: Kernels,
    ctx: Context,
    d_input_nv12: cuda.Buffer(u8),
) !void {
    const d_bgr = try cuda.alloc(f32, ctx.ctx_pitch * ctx.height * 3 * ctx.batch);
    defer d_bgr.free();

    const d_resized_bgr = try cuda.alloc(f32, ctx.dst_width * ctx.dst_height * 3 * ctx.batch);
    defer d_resized_bgr.free();

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    try start.record(.default);
    try cuda.synchronize();
    for (0..test_loop) |_| {
        try nv12_to_bgr_planar.nv12ToBgrPlanarBatch(
            kernels.nv12_to_bgr,
            d_input_nv12,
            ctx.ctx_pitch,
            d_bgr,
            ctx.ctx_pitch * @sizeOf(f32),
            ctx.width,
            ctx.height,
            ctx.batch,
        );
    }
    try stop.record(.default);
    try stop.synchronize();

    var elapsed_time = try start.elapsed(stop);
    printTiming(
        convert_nv12_format,
        ctx,
        .{ ctx.width, ctx.height, ctx.width, ctx.height },
        elapsed_time,
    );

    try start.record(.default);
    for (0..test_loop) |_| {
        try bgr_resize.resizeBgrPlanarBatch(
            kernels.resize_bgr,
            d_bgr,
            ctx.ctx_pitch,
            ctx.width,
            ctx.height,
            d_resized_bgr,
            ctx.dst_width,
            ctx.dst_width,
            ctx.dst_height,
            ctx.batch,
            .{},
            false,
        );
    }
    try stop.record(.default);
    try stop.synchronize();

    elapsed_time = try start.elapsed(stop);
    printTiming(
        resize_bgr_format,
        ctx,
        .{ ctx.width, ctx.height, ctx.dst_width, ctx.dst_height },
        elapsed_time,
    );

    var filename_buffer: [40]u8 = undefined;
    const filename = try std.fmt.bufPrint(&filename_buffer, "resized_bgr_{d}x{d}", .{
        ctx.dst_width,
        ctx.dst_height,
    });
    try utils.dumpBgr(
        gpa,
        io,
        d_resized_bgr,
        ctx.dst_pitch,
        ctx.dst_width,
        ctx.dst_height,
        ctx.batch,
        "t2",
        filename,
    );
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var ctx = (try parseCmdLine(args)) orelse std.process.exit(1);

    const device = try cuda.init(@intCast(ctx.device));
    defer device.deinit();

    const resize_nv12_module = try cuda.Module.load(@embedFile("nv12_resize_kernel.ptx"));
    defer resize_nv12_module.unload();
    const resize_bgr_module = try cuda.Module.load(@embedFile("bgr_resize_kernel.ptx"));
    defer resize_bgr_module.unload();
    const nv12_to_bgr_module = try cuda.Module.load(@embedFile("nv12_to_bgr_planar_kernel.ptx"));
    defer nv12_to_bgr_module.unload();
    const kernels: Kernels = .{
        .resize_nv12 = try resize_nv12_module.function("resizeNv12BatchKernel"),
        .resize_bgr = try resize_bgr_module.function("resizeBgrPlanarBatchKernel"),
        .nv12_to_bgr = try nv12_to_bgr_module.function("nv12ToBgrPlanarBatchKernel"),
    };

    const ctx_alignment = 32;
    ctx.ctx_pitch = std.mem.alignForward(u32, ctx.width, ctx_alignment);

    ctx.ctx_heights = @intFromFloat(@ceil(@as(f32, @floatFromInt(ctx.height)) * 3.0 / 2.0));

    const d_input_nv12 = try cuda.alloc(u8, ctx.ctx_pitch * ctx.ctx_heights * ctx.batch);
    defer d_input_nv12.free();
    loadNv12Frame(gpa, init.io, ctx, d_input_nv12) catch {
        std.log.err("failed to load batch data!", .{});
        std.process.exit(1);
    };

    std.log.info("TEST#1:", .{});
    try nv12ResizeAndNv12ToBgr(gpa, init.io, kernels, ctx, d_input_nv12);

    std.log.info("TEST#2:", .{});
    try nv12ToBgrAndBgrResize(gpa, init.io, kernels, ctx, d_input_nv12);
}
