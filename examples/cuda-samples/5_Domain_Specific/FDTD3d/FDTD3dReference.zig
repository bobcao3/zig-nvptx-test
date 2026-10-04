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

//! The naive host implementation of the finite differences time domain method, which the sample
//! uses as the reference for the GPU result.

const std = @import("std");

pub fn generateRandomData(
    data: []f32,
    random: std.Random,
    lower_bound: f32,
    upper_bound: f32,
) void {
    for (data) |*element| element.* = lower_bound + random.float(f32) * (upper_bound - lower_bound);
}

/// Runs `timesteps` steps of the stencil over `input` and writes the result into `output`. The
/// volume is `dimx` x `dimy` x `dimz` elements with a halo of `radius` elements on every side.
pub fn fdtdReference(
    gpa: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    coeff: []const f32,
    dimx: usize,
    dimy: usize,
    dimz: usize,
    radius: usize,
    timesteps: usize,
) !void {
    const outer_dimx = dimx + 2 * radius;
    const outer_dimy = dimy + 2 * radius;
    const outer_dimz = dimz + 2 * radius;
    const stride_y = outer_dimx;
    const stride_z = stride_y * outer_dimy;

    std.log.info(" calloc intermediate", .{});
    const intermediate = try gpa.alloc(f32, outer_dimx * outer_dimy * outer_dimz);
    defer gpa.free(intermediate);
    @memset(intermediate, 0);

    // Decide which buffer to use first. The result has to end up in `output`.
    var bufsrc = input;
    var bufdst: []f32, var bufdstnext: []f32 = if (timesteps % 2 == 0)
        .{ intermediate, output }
    else
        .{ output, intermediate };

    std.log.info(" Host FDTD loop", .{});

    for (0..timesteps) |it| {
        std.log.debug("\tt = {d}", .{it});

        var index: usize = 0;
        for (0..outer_dimz) |iz| {
            for (0..outer_dimy) |iy| {
                for (0..outer_dimx) |ix| {
                    const inside = ix >= radius and ix < dimx + radius and
                        iy >= radius and iy < dimy + radius and
                        iz >= radius and iz < dimz + radius;
                    if (inside) {
                        var value = bufsrc[index] * coeff[0];

                        for (1..radius + 1) |ir| {
                            // Horizontal.
                            value += coeff[ir] * (bufsrc[index + ir] + bufsrc[index - ir]);
                            // Vertical.
                            value += coeff[ir] * (bufsrc[index + ir * stride_y] + bufsrc[index - ir * stride_y]);
                            // In front and behind.
                            value += coeff[ir] * (bufsrc[index + ir * stride_z] + bufsrc[index - ir * stride_z]);
                        }

                        bufdst[index] = value;
                    } else {
                        bufdst[index] = bufsrc[index];
                    }
                    index += 1;
                }
            }
        }

        // Rotate the buffers.
        const tmp = bufdst;
        bufdst = bufdstnext;
        bufdstnext = tmp;
        bufsrc = tmp;
    }
}

pub fn compareData(
    output: []const f32,
    reference: []const f32,
    dimx: usize,
    dimy: usize,
    dimz: usize,
    radius: usize,
    tolerance: f32,
) bool {
    var index: usize = 0;
    for (0..dimz + 2 * radius) |iz| {
        for (0..dimy + 2 * radius) |iy| {
            for (0..dimx + 2 * radius) |ix| {
                const inside = ix >= radius and ix < dimx + radius and
                    iy >= radius and iy < dimy + radius and
                    iz >= radius and iz < dimz + radius;
                if (inside) {
                    const difference = @abs(reference[index] - output[index]);
                    const relative_error = if (reference[index] != 0) difference / reference[index] else difference;

                    if (relative_error > tolerance) {
                        std.log.err("Data error at point ({d},{d},{d})\t{d:.6} instead of {d:.6}", .{
                            @as(isize, @intCast(ix)) - @as(isize, @intCast(radius)),
                            @as(isize, @intCast(iy)) - @as(isize, @intCast(radius)),
                            @as(isize, @intCast(iz)) - @as(isize, @intCast(radius)),
                            output[index],
                            reference[index],
                        });
                        return false;
                    }
                }
                index += 1;
            }
        }
    }
    return true;
}
