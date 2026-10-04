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

//! A simple program demonstrating why atomic operations are needed, using the atomic
//! intrinsics atomicAdd, atomicMax, and atomicCAS.
//!
//! 1,000,000 work items write into an array of only 10 integers (each work item maps to a slot
//! with index % ARRAY_SIZE), so ~100,000 work items update every element at the same time. A
//! plain update like g[i] = g[i] + 1 is three separate steps: read the value, modify it, write it
//! back. When two work items interleave (both read the same old value, both write back the same
//! result), one update silently overwrites the other and is lost. This is a race condition: the
//! result depends on timing, not program logic, and changes every run.

const std = @import("std");
const cuda = @import("cuda");

const num_threads = 1000000;
const block_width = 1000;
const array_size = 10;

const Operation = enum { add, max, cas };

const Variant = struct { label: []const u8, kernel: [:0]const u8, exact: bool };

const sections = std.enums.EnumArray(Operation, [4]Variant).init(.{
    .add = .{
        .{ .label = "non-atomic", .kernel = "increment", .exact = false },
        .{ .label = "atomicAdd()", .kernel = "incrementAtomic", .exact = true },
        .{ .label = "cuda::std::atomic_ref::fetch_add()", .kernel = "incrementAtomicStd", .exact = true },
        .{ .label = "cuda::atomic_ref::fetch_add()", .kernel = "incrementAtomicCuda", .exact = true },
    },
    .max = .{
        .{ .label = "non-atomic", .kernel = "max", .exact = false },
        .{ .label = "atomicMax()", .kernel = "maxAtomic", .exact = true },
        .{ .label = "cuda::std::atomic_ref::compare_exchange_weak()", .kernel = "maxAtomicStd", .exact = true },
        .{ .label = "cuda::atomic_ref::compare_exchange_weak()", .kernel = "maxAtomicCuda", .exact = true },
    },
    .cas = .{
        .{ .label = "non-atomic", .kernel = "cas", .exact = false },
        .{ .label = "atomicCAS()", .kernel = "casAtomic", .exact = true },
        .{ .label = "cuda::std::atomic_ref::compare_exchange_weak()", .kernel = "casAtomicStd", .exact = true },
        .{ .label = "cuda::atomic_ref::compare_exchange_weak()", .kernel = "casAtomicCuda", .exact = true },
    },
});

fn printArray(writer: *std.Io.Writer, array: []const i32) !void {
    try writer.writeAll("{ ");
    for (array) |element| try writer.print("{d} ", .{element});
    try writer.writeAll("}");
}

fn matchesExpected(operation: Operation, array: []const i32) bool {
    for (array, 0..) |element, i| {
        const expected: i32 = switch (operation) {
            .add, .cas => num_threads / array_size,
            .max => num_threads - array_size + @as(i32, @intCast(i)),
        };
        if (element != expected) return false;
    }
    return true;
}

pub fn main() !void {
    std.log.info(
        "=== Atomic vs. non-atomic operations (intrinsics, cuda::std::atomic_ref, cuda::atomic_ref) ===",
        .{},
    );

    const dev_id = 0;
    const device = try cuda.init(dev_id);
    defer device.deinit();

    const capability = try device.computeCapability();
    const sm_count = try device.attribute(.multiprocessor_count);
    std.log.info(
        "GPU Device {d}: with compute capability {d}.{d} and Number of SMs {d}",
        .{ dev_id, capability.major, capability.minor, sm_count },
    );

    const module = try cuda.Module.load(@embedFile("simpleAtomicIntrinsics_kernel.ptx"));
    defer module.unload();

    // Enough work groups to cover all work items, rounding up if not evenly divisible.
    const num_blocks = std.math.divCeil(u32, num_threads, block_width) catch unreachable;
    std.log.info(
        "{d} total threads in {d} blocks writing into {d} array elements",
        .{ num_threads, num_blocks, array_size },
    );

    var h_array: [array_size]i32 = undefined;
    const d_array = try cuda.alloc(i32, array_size);
    defer d_array.free();

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    var all_exact = true;
    for (std.enums.values(Operation)) |operation| {
        switch (operation) {
            .add => std.log.info("[add] expected: every element = {d}", .{num_threads / array_size}),
            .max => std.log.info("[max] expected: element i = {d} + i", .{num_threads - array_size}),
            .cas => std.log.info("[CAS] expected: every element = {d}", .{num_threads / array_size}),
        }
        const label_width: usize = if (operation == .add) 36 else 46;

        for (sections.get(operation)) |variant| {
            const kernel = try module.function(variant.kernel);
            try d_array.memset(0);

            try start.record(.default);
            try kernel.launch(.{ .grid = .{ .x = num_blocks }, .block = .{ .x = block_width } }, .{d_array});
            try stop.record(.default);

            try stop.synchronize();
            const elapsed_ms = try start.elapsed(stop);
            try d_array.copyTo(&h_array);

            var line_buffer: [512]u8 = undefined;
            var line: std.Io.Writer = .fixed(&line_buffer);
            try line.writeAll(variant.label);
            for (variant.label.len..label_width) |_| try line.writeByte(' ');
            try line.print(" ({d} ms): ", .{elapsed_ms});
            try printArray(&line, &h_array);
            std.log.info("{s}", .{line.buffered()});
            if (variant.exact and !matchesExpected(operation, &h_array)) all_exact = false;
        }
    }

    std.log.info("The non-atomic kernels lose updates when threads race on the same", .{});
    std.log.info("element; the atomic kernels match the expected values exactly.", .{});
    if (!all_exact) {
        std.log.err("An atomic kernel did not match the expected values.", .{});
        std.process.exit(1);
    }
}
