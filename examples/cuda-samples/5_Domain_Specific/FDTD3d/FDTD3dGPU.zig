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

//! The GPU implementation of the finite differences time domain method.

const std = @import("std");
const cuda = @import("cuda");
const constants = @import("constants.zig");

/// Runs `timesteps` steps of the stencil on `input` with the GPU and writes the result into
/// `output`. `block_size` is the requested number of work items in a work group, if any.
pub fn fdtdGpu(
    device: cuda.Device,
    output: []f32,
    input: []const f32,
    coeff: []const f32,
    dimx: u32,
    dimy: u32,
    dimz: u32,
    radius: u32,
    timesteps: u32,
    block_size: ?u32,
) !void {
    const outer_dimx = dimx + 2 * radius;
    const outer_dimy = dimy + 2 * radius;
    const outer_dimz = dimz + 2 * radius;
    const volume_size: usize = @as(usize, outer_dimx) * outer_dimy * outer_dimz;

    // Ensure that the inner data starts on a 128 B boundary.
    const padding: usize = (128 / @sizeOf(f32)) - radius;
    const padded_volume_size = volume_size + padding;

    if (radius != constants.radius) {
        std.log.err("radius is invalid, must be {d} - see kernel for details.", .{constants.radius});
        return error.InvalidRadius;
    }

    const module = try cuda.Module.load(@embedFile("FDTD3dGPU_kernel.ptx"));
    defer module.unload();
    const finite_differences_kernel = try module.function("finiteDifferencesKernel");

    const buffer_out = try cuda.alloc(f32, padded_volume_size);
    defer buffer_out.free();
    const buffer_in = try cuda.alloc(f32, padded_volume_size);
    defer buffer_in.free();

    // Check for a command-line specified block size.
    var user_block_size: u32 = constants.block_size_max;
    if (block_size) |requested| {
        // Constrain to a multiple of `block_dim_x`, then within the allowed bounds.
        user_block_size = requested / constants.block_dim_x * constants.block_dim_x;
        user_block_size = @min(@max(user_block_size, constants.block_size_min), constants.block_size_max);
    }

    // Check the device limit on the number of work items in a work group.
    user_block_size = @min(user_block_size, @as(u32, @intCast(try device.attribute(.max_threads_per_block))));

    const dim_block: cuda.Dim3 = .{
        .x = constants.block_dim_x,
        .y = @min(user_block_size / constants.block_dim_x, constants.block_dim_max_y),
    };
    const dim_grid: cuda.Dim3 = .{
        .x = std.math.divCeil(u32, dimx, dim_block.x) catch unreachable,
        .y = std.math.divCeil(u32, dimy, dim_block.y) catch unreachable,
    };
    std.log.info(" set block size to {d}x{d}", .{ dim_block.x, dim_block.y });
    std.log.info(" set grid size to {d}x{d}", .{ dim_grid.x, dim_grid.y });

    if (dim_block.x < radius or dim_block.y < radius) {
        std.log.err("invalid block size, x ({d}) and y ({d}) must be >= radius ({d}).", .{
            dim_block.x, dim_block.y, radius,
        });
        return error.InvalidBlockSize;
    }

    // The buffers that the kernel reads and writes start `padding` elements into the allocations.
    var buffer_src: cuda.Buffer(f32) = .{
        .address = buffer_in.address + padding * @sizeOf(f32),
        .len = volume_size,
    };
    var buffer_dst: cuda.Buffer(f32) = .{
        .address = buffer_out.address + padding * @sizeOf(f32),
        .len = volume_size,
    };

    // Copy the input to the device input buffer.
    try buffer_src.copyFrom(input);

    // Copy the input to the device output buffer. Only the halo is needed.
    try buffer_dst.copyFrom(input);

    // Copy the coefficients to the device coefficient buffer.
    const stencil = try module.global(f32, "stencil");
    try stencil.copyFrom(coeff);

    std.log.info(" GPU FDTD loop", .{});

    for (0..timesteps) |it| {
        std.log.debug("\tt = {d} launch kernel", .{it});

        try finite_differences_kernel.launch(
            .{ .grid = dim_grid, .block = dim_block },
            .{ buffer_dst, buffer_src, @as(i32, @intCast(dimx)), @as(i32, @intCast(dimy)), @as(i32, @intCast(dimz)) },
        );

        // Toggle the buffers.
        std.mem.swap(cuda.Buffer(f32), &buffer_dst, &buffer_src);
    }

    // Wait for the kernel to complete.
    try cuda.synchronize();

    // Read the result back. The result is in `buffer_src` after the final toggle.
    try buffer_src.copyTo(output);
}
