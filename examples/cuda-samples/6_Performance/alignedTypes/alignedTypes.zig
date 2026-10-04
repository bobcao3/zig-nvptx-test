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

//! This is a simple test showing the huge access speed gap between aligned and misaligned
//! structures (those with and without an alignment declaration). It measures the per-element copy
//! throughput for aligned and misaligned structures on big chunks of data.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("alignedTypes_common.zig");

/// The size of the memory chunk in bytes, reused for each test.
const mem_size = 50000000;
const num_iterations = 32;

/// Compares only the "packed" part of each structure, which holds the user data. The padding
/// bytes of a structure hold no user data.
fn testCpu(
    comptime Element: type,
    h_odata: []const Element,
    h_idata: []const Element,
    packed_element_size: usize,
) bool {
    for (h_idata, h_odata) |src, dst| {
        const src_bytes = std.mem.asBytes(&src)[0..packed_element_size];
        const dst_bytes = std.mem.asBytes(&dst)[0..packed_element_size];
        if (!std.mem.eql(u8, src_bytes, dst_bytes)) return false;
    }
    return true;
}

const Test = struct {
    module: cuda.Module,
    d_idata: cuda.Buffer(u8),
    d_odata: cuda.Buffer(u8),
    h_idata: []align(16) const u8,
    h_odata: []align(16) u8,
    start: cuda.Event,
    stop: cuda.Event,

    /// Copies the memory with the kernel of `kind` and returns whether the copy is correct.
    fn run(self: Test, comptime kind: common.Kind, packed_element_size: usize) !bool {
        const Element = kind.Element();
        const memory_size = self.h_idata.len;
        const total_mem_size_aligned = memory_size - memory_size % @sizeOf(Element);
        const num_elements = memory_size / @sizeOf(Element);
        const test_kernel = try self.module.function("testKernel_" ++ @tagName(kind));

        // Cleans the output buffer before the test.
        try self.d_odata.memset(0);
        try cuda.synchronize();
        try self.start.record(.default);

        for (0..num_iterations) |_| {
            try test_kernel.launch(.{ .grid = .{ .x = 64 }, .block = .{ .x = 256 } }, .{
                self.d_odata,
                self.d_idata,
                @as(i32, @intCast(num_elements)),
            });
        }

        try self.stop.record(.default);
        try self.stop.synchronize();
        const gpu_time = try self.start.elapsed(self.stop) / num_iterations;
        const throughput = @as(f64, @floatFromInt(total_mem_size_aligned)) /
            (@as(f64, gpu_time) * 0.001 * 1073741824.0);
        std.log.info("Avg. time: {d:.6} ms / Copy throughput: {d:.6} GB/s.", .{
            gpu_time,
            throughput,
        });

        // Reads back the GPU results and runs the validation.
        try self.d_odata.copyTo(self.h_odata);
        const used_bytes = num_elements * @sizeOf(Element);
        const elements_in = std.mem.bytesAsSlice(Element, self.h_idata[0..used_bytes]);
        const elements_out = std.mem.bytesAsSlice(Element, self.h_odata[0..used_bytes]);
        const flag = testCpu(Element, elements_out, elements_in, packed_element_size);

        if (flag) std.log.info("\tTEST OK", .{}) else std.log.err("\tTEST FAILURE", .{});
        return flag;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("[alignedTypes] - Starting...", .{});

    const device = try cuda.init(0);
    defer device.deinit();

    var name_buffer: [256]u8 = undefined;
    const capability = try device.computeCapability();
    const multiprocessor_count = try device.attribute(.multiprocessor_count);
    const cores = cuda.coresPerMultiprocessor(capability.major, capability.minor);
    std.log.info("[{s}] has {d} MP(s) x {d} (Cores/MP) = {d} (Cores)", .{
        try device.name(&name_buffer),
        multiprocessor_count,
        cores,
        cores * multiprocessor_count,
    });

    // Anything that is less than 192 cores has a scaled down workload.
    const scale_factor = @max(192.0 / @as(f32, @floatFromInt(cores * multiprocessor_count)), 1.0);

    // The memory size is a multiple of 256 bytes.
    const memory_size = @as(usize, @intFromFloat(mem_size / scale_factor)) & 0xffffff00;

    std.log.info("> Compute scaling value = {d:.2}", .{scale_factor});
    std.log.info("> Memory Size = {d}", .{memory_size});

    std.log.info("Allocating memory...", .{});
    const h_idata = try gpa.alignedAlloc(u8, .@"16", memory_size);
    defer gpa.free(h_idata);
    const h_odata = try gpa.alignedAlloc(u8, .@"16", memory_size);
    defer gpa.free(h_odata);
    const d_idata = try cuda.alloc(u8, memory_size);
    defer d_idata.free();
    const d_odata = try cuda.alloc(u8, memory_size);
    defer d_odata.free();

    std.log.info("Generating host input data array...", .{});
    for (h_idata, 0..) |*byte, i| byte.* = @as(u8, @truncate(i)) +% 1;

    std.log.info("Uploading input data to GPU memory...", .{});
    try d_idata.copyFrom(h_idata);

    const module = try cuda.Module.load(@embedFile("alignedTypes_kernel.ptx"));
    defer module.unload();
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    const tests: Test = .{
        .module = module,
        .d_idata = d_idata,
        .d_odata = d_odata,
        .h_idata = h_idata,
        .h_odata = h_odata,
        .start = start,
        .stop = stop,
    };

    var total_failures: u32 = 0;

    std.log.info("Testing misaligned types...", .{});
    inline for (.{
        .{ .uint8, "uint8", 1 },
        .{ .uint16, "uint16", 2 },
        .{ .rgba8_misaligned, "RGBA8_misaligned", 4 },
        .{ .la32_misaligned, "LA32_misaligned", 8 },
        .{ .rgb32_misaligned, "RGB32_misaligned", 12 },
        .{ .rgba32_misaligned, "RGBA32_misaligned", 16 },
    }) |entry| {
        std.log.info("{s}...", .{entry[1]});
        if (!try tests.run(entry[0], entry[2])) total_failures += 1;
    }

    std.log.info("Testing aligned types...", .{});
    inline for (.{
        .{ .rgba8, "RGBA8", 4 },
        .{ .i32, "I32", 4 },
        .{ .la32, "LA32", 8 },
        .{ .rgb32, "RGB32", 12 },
        .{ .rgba32, "RGBA32", 16 },
        .{ .rgba32_2, "RGBA32_2", 32 },
    }) |entry| {
        std.log.info("{s}...", .{entry[1]});
        if (!try tests.run(entry[0], entry[2])) total_failures += 1;
    }

    std.log.info("[alignedTypes] -> Test Results: {d} Failures", .{total_failures});
    std.log.info("Shutting down...", .{});

    if (total_failures != 0) {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }
    std.log.info("Test passed", .{});
}
