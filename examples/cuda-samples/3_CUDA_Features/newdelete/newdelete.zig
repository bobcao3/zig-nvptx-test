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

//! This sample demonstrates dynamic global memory allocation through device `malloc` and `free`
//! and virtual function declarations available with CUDA 4.0.

const std = @import("std");
const cuda = @import("cuda");

fn checkResult(gpa: std.mem.Allocator, d_result: cuda.Buffer(i32), n: usize) !bool {
    const h_result = try gpa.alloc(i32, n);
    defer gpa.free(h_result);

    try d_result.copyTo(h_result);
    std.mem.sort(i32, h_result, {}, std.sort.asc(i32));

    var success = true;
    var is_test = false;

    var value: i32 = 0;

    for (h_result) |element| {
        if (element != -1) {
            is_test = true;
        }

        if (is_test) {
            if (value != element) success = false;
            value += 1;
        }
    }

    return success;
}

fn testContainer(
    gpa: std.mem.Allocator,
    module: cuda.Module,
    d_container: cuda.Buffer(u64),
    blocks: u32,
    threads: u32,
) !bool {
    const container_fill = try module.function("containerFill");
    const container_consume = try module.function("containerConsume");
    const container_delete = try module.function("containerDelete");

    const d_result = try cuda.alloc(i32, blocks * threads);
    defer d_result.free();

    try container_fill.launch(.{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, .{d_container});
    try container_consume.launch(
        .{ .grid = .{ .x = blocks }, .block = .{ .x = threads } },
        .{ d_container, d_result },
    );
    try container_delete.launch(.{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, .{d_container});
    try cuda.synchronize();

    return checkResult(gpa, d_result, blocks * threads);
}

fn testPlacementNew(gpa: std.mem.Allocator, module: cuda.Module, threads: u32) !bool {
    const placement_new = try module.function("placementNew");
    const d_result = try cuda.alloc(i32, threads);
    defer d_result.free();

    try placement_new.launch(.{ .grid = .{ .x = 1 }, .block = .{ .x = threads } }, .{d_result});
    try cuda.synchronize();

    return checkResult(gpa, d_result, threads);
}

fn testComplexType(gpa: std.mem.Allocator, module: cuda.Module, threads: u32) !bool {
    const complex_vector = try module.function("complexVector");
    const d_result = try cuda.alloc(i32, threads);
    defer d_result.free();

    try complex_vector.launch(.{ .grid = .{ .x = 1 }, .block = .{ .x = threads } }, .{d_result});
    try cuda.synchronize();

    return checkResult(gpa, d_result, threads);
}

pub fn main(init: std.process.Init) !void {
    std.log.info("newdelete Starting...", .{});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("newdelete_kernel.ptx"));
    defer module.unload();

    // Set the heap size for device size new/delete to 128 MB.
    try cuda.setLimit(.malloc_heap_size, 128 * (1 << 20));

    var test_passed: u32 = 0;

    {
        const d_container = try cuda.alloc(u64, 1);
        defer d_container.free();

        std.log.info(" > Container = Vector test ", .{});
        const vector_create = try module.function("vectorCreate");
        try vector_create.launch(
            .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } },
            .{ d_container, @as(i32, 128 * 128) },
        );
        const passed = try testContainer(init.gpa, module, d_container, 128, 128);
        if (passed) std.log.info("OK", .{}) else std.log.err("NOT OK", .{});
        test_passed += @intFromBool(passed);
    }

    std.log.info(" > Container = Vector, using placement new on SMEM buffer test ", .{});
    const placement_passed = try testPlacementNew(init.gpa, module, 1024);
    if (placement_passed) std.log.info("OK", .{}) else std.log.err("NOT OK", .{});
    test_passed += @intFromBool(placement_passed);

    std.log.info(" > Container = Vector, with user defined datatype test ", .{});
    const complex_passed = try testComplexType(init.gpa, module, 1024);
    if (complex_passed) std.log.info("OK", .{}) else std.log.err("NOT OK", .{});
    test_passed += @intFromBool(complex_passed);

    std.log.info("Test Summary: {d}/3 succesfully run", .{test_passed});

    if (test_passed != 3) std.process.exit(1);
}
