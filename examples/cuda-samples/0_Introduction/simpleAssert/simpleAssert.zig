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

const sample_name = "simpleAssert";

pub fn main() !void {
    const n_blocks = 2;
    const n_threads = 32;

    std.log.info("{s} starting...", .{sample_name});

    const dev_id = 0;
    const device = try cuda.init(dev_id);
    defer device.deinit();

    const capability = try device.computeCapability();
    const sm_count = try device.attribute(.multiprocessor_count);
    std.log.info(
        "GPU Device {d}: with compute capability {d}.{d} and Number of SMs {d}",
        .{ dev_id, capability.major, capability.minor, sm_count },
    );

    const module = try cuda.Module.load(@embedFile("simpleAssert_kernel.ptx"));
    defer module.unload();
    const simple_assert_kernel = try module.function("simpleAssertKernel");

    std.log.info("Launch kernel to generate assertion failures", .{});
    try simple_assert_kernel.launch(
        .{ .grid = .{ .x = n_blocks }, .block = .{ .x = n_threads } },
        .{@as(i32, 60)},
    );

    std.log.info("-- Begin assert output", .{});
    const synchronized = cuda.synchronize();
    std.log.info("-- End assert output", .{});

    const test_result = if (synchronized) |_| false else |err| switch (err) {
        error.Assert, error.LaunchFailed => blk: {
            std.log.info("Device assert failed as expected, CUDA error is: {t}", .{err});
            break :blk true;
        },
        else => false,
    };

    if (test_result) {
        std.log.info("{s} completed, returned OK", .{sample_name});
    } else {
        std.log.err("{s} completed, returned ERROR!", .{sample_name});
    }
    if (!test_result) std.process.exit(1);
}
