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

const std = @import("std");
const cuda = @import("cuda");

pub fn main() !void {
    const dev_id = 0;
    const device = try cuda.init(dev_id);
    defer device.deinit();

    const capability = try device.computeCapability();
    const sm_count = try device.attribute(.multiprocessor_count);
    std.log.info(
        "GPU Device {d}: with compute capability {d}.{d} and Number of SMs {d}",
        .{ dev_id, capability.major, capability.minor, sm_count },
    );

    const module = try cuda.Module.load(@embedFile("simplePrintf_kernel.ptx"));
    defer module.unload();
    const simple_printf_kernel = try module.function("simplePrintfKernel");

    std.log.info("printf() is called. Output:", .{});

    // A two-dimensional grid and three-dimensional work groups.
    try simple_printf_kernel.launch(
        .{ .grid = .{ .x = 2, .y = 2 }, .block = .{ .x = 2, .y = 2, .z = 2 } },
        .{@as(i32, 10)},
    );
    try cuda.synchronize();
}
