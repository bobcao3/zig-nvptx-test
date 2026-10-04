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

//! This sample applies a finite differences time domain progression stencil on a 3D surface.

const std = @import("std");
const cuda = @import("cuda");
const fdtd_gpu = @import("FDTD3dGPU.zig");
const reference = @import("FDTD3dReference.zig");

// The values are set to give reasonable runtimes. They can be changed, but running very large
// dimensions can take a very long time.
const k_dim_min = 96;
const k_dim_max = 376;
const k_dim_qa = 248;

// The radius is exactly 4 because the kernel uses a constant.
const k_radius_min = 4;
const k_radius_max = 4;
const k_radius_default = 4;

const k_timesteps_min = 1;
const k_timesteps_max = 10;
const k_timesteps_default = 5;

/// Returns the value of the command line argument `--<name>=<value>`.
fn flagValue(args: []const [:0]const u8, name: []const u8) !?i32 {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len == arg.len or !std.mem.startsWith(u8, flag, name)) continue;
        if (flag.len > name.len and flag[name.len] == '=') {
            return try std.fmt.parseInt(i32, flag[name.len + 1 ..], 10);
        }
    }
    return null;
}

fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args[1..]) |arg| {
        const flag = std.mem.trimStart(u8, arg, "-");
        if (flag.len != arg.len and std.mem.startsWith(u8, flag, name)) return true;
    }
    return false;
}

fn showHelp(args: []const [:0]const u8) void {
    std.log.info("{s}", .{args[0]});
    std.log.info(
        \\Syntax:
        \\    --device=<device>   Specify device to use for execution
        \\    --dimx=<N>          Specify number of elements in x direction (excluding halo)
        \\    --dimy=<N>          Specify number of elements in y direction (excluding halo)
        \\    --dimz=<N>          Specify number of elements in z direction (excluding halo)
        \\    --radius=<N>        Specify radius of stencil
        \\    --timesteps=<N>     Specify number of timesteps
        \\    --block-size=<N>    Specify number of threads per block
        \\
        \\    --noprompt          Skip prompt before exit
    , .{});
}

fn runTest(init: std.process.Init, args: []const [:0]const u8) !bool {
    const gpa = init.gpa;
    const lower_bound = 0.0;
    const upper_bound = 1.0;

    const device = try cuda.init(@intCast(try flagValue(args, "device") orelse 0));
    defer device.deinit();

    // Determine the default dimensions. Every device with at least 1 GiB of memory allows the
    // maximum dimension, which is the default that the original computes from the device memory.
    std.log.info("Set-up, based upon target device GMEM size...", .{});
    var default_dim: i32 = k_dim_max;

    // For QA testing, override the default volume size.
    if (hasFlag(args, "qatest")) default_dim = @min(default_dim, k_dim_qa);

    var dimx = default_dim;
    var dimy = default_dim;
    var dimz = default_dim;
    var radius: i32 = k_radius_default;
    var timesteps: i32 = k_timesteps_default;

    // Parse the command line arguments.
    if (try flagValue(args, "dimx")) |value| dimx = std.math.clamp(value, k_dim_min, k_dim_max);
    if (try flagValue(args, "dimy")) |value| dimy = std.math.clamp(value, k_dim_min, k_dim_max);
    if (try flagValue(args, "dimz")) |value| dimz = std.math.clamp(value, k_dim_min, k_dim_max);
    if (try flagValue(args, "radius")) |value| radius = std.math.clamp(value, k_radius_min, k_radius_max);
    if (try flagValue(args, "timesteps")) |value| {
        timesteps = std.math.clamp(value, k_timesteps_min, k_timesteps_max);
    }
    const block_size: ?u32 = if (try flagValue(args, "block-size")) |value| @intCast(@max(value, 0)) else null;

    // Determine the volume size.
    const outer_dimx: usize = @intCast(dimx + 2 * radius);
    const outer_dimy: usize = @intCast(dimy + 2 * radius);
    const outer_dimz: usize = @intCast(dimz + 2 * radius);
    const volume_size = outer_dimx * outer_dimy * outer_dimz;

    const host_output = try gpa.alloc(f32, volume_size);
    defer gpa.free(host_output);
    @memset(host_output, 0);
    const input = try gpa.alloc(f32, volume_size);
    defer gpa.free(input);
    const coeff = try gpa.alloc(f32, @intCast(radius + 1));
    defer gpa.free(coeff);

    @memset(coeff, 0.1);

    std.log.info(" generateRandomData", .{});
    var prng: std.Random.DefaultPrng = .init(0);
    reference.generateRandomData(input, prng.random(), lower_bound, upper_bound);
    std.log.info(
        "FDTD on {d} x {d} x {d} volume with symmetric filter radius {d} for {d} timesteps...",
        .{ dimx, dimy, dimz, radius, timesteps },
    );

    std.log.info("fdtdReference...", .{});
    try reference.fdtdReference(
        gpa,
        host_output,
        input,
        coeff,
        @intCast(dimx),
        @intCast(dimy),
        @intCast(dimz),
        @intCast(radius),
        @intCast(timesteps),
    );
    std.log.info("fdtdReference complete", .{});

    const device_output = try gpa.alloc(f32, volume_size);
    defer gpa.free(device_output);
    @memset(device_output, 0);

    std.log.info("fdtdGPU...", .{});
    try fdtd_gpu.fdtdGpu(
        device,
        device_output,
        input,
        coeff,
        @intCast(dimx),
        @intCast(dimy),
        @intCast(dimz),
        @intCast(radius),
        @intCast(timesteps),
        block_size,
    );
    std.log.info("fdtdGPU complete", .{});

    const tolerance = 0.0001;
    std.log.info("CompareData (tolerance {d:.6})...", .{tolerance});
    return reference.compareData(
        device_output,
        host_output,
        @intCast(dimx),
        @intCast(dimy),
        @intCast(dimz),
        @intCast(radius),
        tolerance,
    );
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    if (hasFlag(args, "help")) {
        std.log.info("Displaying help on console", .{});
        showHelp(args);
        return;
    }
    if (!try runTest(init, args)) std.process.exit(1);
}
