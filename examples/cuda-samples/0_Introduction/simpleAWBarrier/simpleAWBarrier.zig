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

const std = @import("std");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} starting...", .{args[0]});

    const device_id = 0;
    const device = try cuda.init(device_id);
    defer device.deinit();

    const capability = try device.computeCapability();
    if (capability.major < 7) {
        std.log.warn("simpleAWBarrier requires SM 7.0 or higher.  Exiting...", .{});
        std.process.exit(2);
    }

    if (try device.attribute(.cooperative_launch) == 0) {
        std.log.warn("Selected GPU ({d}) does not support Cooperative Kernel Launch, " ++
            "Waiving the run", .{device_id});
        std.process.exit(2);
    }

    if (!try runNormVecByDotProductAwBarrier(init, device)) {
        std.log.info("{s} completed, returned ERROR!", .{args[0]});
        std.process.exit(1);
    }
    std.log.info("{s} completed, returned OK", .{args[0]});
}

fn runNormVecByDotProductAwBarrier(init: std.process.Init, device: cuda.Device) !bool {
    const size = 10000000;

    const module = try cuda.Module.load(@embedFile("simpleAWBarrier_kernel.ptx"));
    defer module.unload();
    const kernel = try module.function("normVecByDotProductAwBarrier");

    const vec_a = try init.gpa.alloc(f32, size);
    defer init.gpa.free(vec_a);
    const vec_b = try init.gpa.alloc(f32, size);
    defer init.gpa.free(vec_b);

    const d_vec_a = try cuda.alloc(f32, size);
    defer d_vec_a.free();
    const d_vec_b = try cuda.alloc(f32, size);
    defer d_vec_b.free();

    const base_val: f32 = 2.0;
    @memset(vec_a, base_val);
    @memset(vec_b, base_val);

    const stream = try cuda.Stream.create();
    defer stream.destroy();

    try d_vec_a.copyFromAsync(vec_a, stream);
    try d_vec_b.copyFromAsync(vec_b, stream);

    const block_size = (try kernel.maxPotentialBlockSize(0, size)).block_size;

    const smem_size = ((block_size / 32) + 1) * @sizeOf(f64);

    const num_blocks_per_sm = try kernel.maxActiveBlocksPerMultiprocessor(block_size, smem_size);

    const multiprocessor_count: u32 = @intCast(try device.attribute(.multiprocessor_count));

    const min_grid_size = multiprocessor_count * num_blocks_per_sm;
    const d_partial_results = try cuda.alloc(f64, min_grid_size);
    defer d_partial_results.free();

    std.log.info("Launching normVecByDotProductAWBarrier kernel with numBlocks = {d} " ++
        "blockSize = {d}", .{ min_grid_size, block_size });

    try kernel.launchCooperative(.{
        .grid = .{ .x = min_grid_size },
        .block = .{ .x = block_size },
        .shared_memory_bytes = smem_size,
        .stream = stream,
    }, .{ d_vec_a, d_vec_b, d_partial_results, @as(i32, size) });

    try d_vec_a.copyToAsync(vec_a, stream);
    try stream.synchronize();

    const expected_result: f32 = base_val / @sqrt(@as(f32, size) * base_val * base_val);
    var matches: u32 = 0;
    for (vec_a, 0..) |element, i| {
        if (element - expected_result > 0.00001) {
            std.log.info("mismatch at i = {d}", .{i});
            break;
        }
        matches += 1;
    }

    std.log.info("Result = {s}", .{if (matches == size) "PASSED" else "FAILED"});
    return matches == size;
}
