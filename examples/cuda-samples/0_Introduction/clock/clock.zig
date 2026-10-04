// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

//! This example shows how to use the clock function to measure the performance of block of
//! threads of a kernel accurately. Blocks are executed in parallel and out of order. Since
//! there's no synchronization mechanism between blocks, we measure the clock once for each block.
//! The clock samples are written to device memory.

const std = @import("std");
const cuda = @import("cuda");

const num_blocks = 64;
const num_threads = 256;

pub fn main() !void {
    std.log.info("CUDA Clock sample", .{});

    const dev_id = 0;
    const device = try cuda.init(dev_id);
    defer device.deinit();

    const capability = try device.computeCapability();
    const sm_count = try device.attribute(.multiprocessor_count);
    std.log.info(
        "GPU Device {d}: with compute capability {d}.{d} and Number of SMs {d}",
        .{ dev_id, capability.major, capability.minor, sm_count },
    );

    const module = try cuda.Module.load(@embedFile("clock_kernel.ptx"));
    defer module.unload();
    const timed_reduction = try module.function("timedReduction");

    var timer: [num_blocks * 2]i64 = undefined;
    var output: [num_blocks]f32 = undefined;
    var input: [num_threads * 2]f32 = undefined;
    for (&input, 0..) |*element, i| element.* = @floatFromInt(i);

    const dinput = try cuda.dupe(f32, &input);
    defer dinput.free();
    const doutput = try cuda.alloc(f32, num_blocks);
    defer doutput.free();
    const dtimer = try cuda.alloc(i64, num_blocks * 2);
    defer dtimer.free();

    try timed_reduction.launch(
        .{ .grid = .{ .x = num_blocks }, .block = .{ .x = num_threads } },
        .{ dinput, doutput, dtimer },
    );

    try dtimer.copyTo(&timer);
    try doutput.copyTo(&output);

    var avg_elapsed_clocks: f64 = 0;
    for (timer[0..num_blocks], timer[num_blocks..]) |begin, end| {
        avg_elapsed_clocks += @floatFromInt(end - begin);
    }
    avg_elapsed_clocks /= num_blocks;
    std.log.info("Average clocks/block = {d:.6}", .{avg_elapsed_clocks});

    for (output) |block_min| {
        if (block_min != 0) {
            std.log.err("Block minimum {d} is not 0", .{block_min});
            std.process.exit(1);
        }
    }
}
