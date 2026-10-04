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

//! Scalar product of two half-precision vectors with `f16` pairs, computed once with the
//! arithmetic operators and once with fused multiply-add and add on pairs.

const std = @import("std");
const cuda = @import("cuda");

const Half2 = @Vector(2, f16);

const num_of_blocks = 128;
const num_of_threads = 128;

fn generateInput(a: []Half2, random: std.Random) void {
    for (a) |*element| {
        element.* = .{
            @floatFromInt(random.uintLessThan(u32, 4)),
            @floatFromInt(random.uintLessThan(u32, 2)),
        };
    }
}

fn sum(values: []const f32) f32 {
    var total: f32 = 0;
    for (values) |value| total += value;
    return total;
}

pub fn main(init: std.process.Init) !void {
    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.real.now(init.io).toSeconds()));
    const size = num_of_blocks * num_of_threads * 16;

    const device = try cuda.init(0);
    defer device.deinit();

    const compute_capability = try device.computeCapability();
    if (compute_capability.major < 5 or (compute_capability.major == 5 and compute_capability.minor < 3)) {
        std.log.err("ERROR: fp16ScalarProduct requires GPU devices with compute SM 5.3 or higher.", .{});
        return;
    }

    const module = try cuda.Module.load(@embedFile("fp16ScalarProduct_kernel.ptx"));
    defer module.unload();
    const native = try module.function("scalarProductKernelNative");
    const intrinsics = try module.function("scalarProductKernelIntrinsics");

    var dev_vec: [2]cuda.Buffer(Half2) = undefined;
    for (&dev_vec) |*buffer| {
        const host = try init.gpa.alloc(Half2, size);
        defer init.gpa.free(host);
        generateInput(host, prng.random());
        buffer.* = try cuda.dupe(Half2, host);
    }
    defer for (dev_vec) |buffer| buffer.free();

    const results = try init.gpa.alloc(f32, num_of_blocks);
    defer init.gpa.free(results);
    const dev_results = try cuda.alloc(f32, num_of_blocks);
    defer dev_results.free();

    const config: cuda.LaunchConfig = .{ .grid = .{ .x = num_of_blocks }, .block = .{ .x = num_of_threads } };

    try native.launch(config, .{ dev_vec[0], dev_vec[1], dev_results, @as(u64, size) });
    try dev_results.copyTo(results);
    const result_native = sum(results);
    std.log.info("Result native operators\t: {d:.6} ", .{result_native});

    try intrinsics.launch(config, .{ dev_vec[0], dev_vec[1], dev_results, @as(u64, size) });
    try dev_results.copyTo(results);
    const result_intrinsics = sum(results);
    std.log.info("Result intrinsics\t: {d:.6} ", .{result_intrinsics});

    const passed = @abs(result_intrinsics - result_native) < 0.00001;
    if (passed) std.log.info("&&&& fp16ScalarProduct PASSED", .{}) else std.log.err("&&&& fp16ScalarProduct FAILED", .{});
    if (!passed) std.process.exit(1);
}
