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

const loop_num = 50;

fn valCompareAndSwap(ptr: *i32, expected: i32, desired: i32) i32 {
    return @cmpxchgStrong(i32, ptr, expected, desired, .seq_cst, .seq_cst) orelse expected;
}

fn atomicKernelCpu(atom_arr: []i32, no_of_threads: i32) void {
    var i = no_of_threads;
    while (i < 2 * no_of_threads) : (i += 1) {
        for (0..loop_num) |_| {
            _ = @atomicRmw(i32, &atom_arr[0], .Add, 10, .seq_cst);

            _ = @atomicRmw(i32, &atom_arr[1], .Xchg, i, .seq_cst);

            while (true) {
                const expected = @atomicLoad(i32, &atom_arr[2], .seq_cst);
                if (valCompareAndSwap(&atom_arr[2], expected, @max(expected, i)) == expected) break;
            }

            while (true) {
                const expected = @atomicLoad(i32, &atom_arr[3], .seq_cst);
                if (valCompareAndSwap(&atom_arr[3], expected, @min(expected, i)) == expected) break;
            }

            var limit: i32 = 17;
            while (true) {
                const expected = @atomicLoad(i32, &atom_arr[4], .seq_cst);
                const desired = if (expected >= limit) 0 else expected + 1;
                if (valCompareAndSwap(&atom_arr[4], expected, desired) == expected) break;
            }

            limit = 137;
            while (true) {
                const expected = @atomicLoad(i32, &atom_arr[5], .seq_cst);
                const desired = if (expected == 0 or expected > limit) limit else expected - 1;
                if (valCompareAndSwap(&atom_arr[5], expected, desired) == expected) break;
            }

            _ = valCompareAndSwap(&atom_arr[6], i - 1, i);

            _ = @atomicRmw(i32, &atom_arr[7], .And, 2 * i + 7, .seq_cst);

            _ = @atomicRmw(i32, &atom_arr[8], .Or, std.math.shl(i32, 1, i), .seq_cst);

            _ = @atomicRmw(i32, &atom_arr[9], .Xor, i, .seq_cst);
        }
    }
}

/// Computes the reference results for `len` work items and compares them with `test_data`.
fn verify(test_data: []const i32, len: i32) bool {
    var val: i32 = 0;

    for (0..@as(usize, @intCast(len)) * loop_num) |_| val += 10;

    if (val != test_data[0]) {
        std.log.err("atomicAdd failed val = {d} testData = {d}", .{ val, test_data[0] });
        return false;
    }

    var found = false;

    var i: i32 = 0;
    while (i < len) : (i += 1) {
        if (i == test_data[1]) {
            found = true;
            break;
        }
    }

    if (!found) {
        std.log.err("atomicExch failed", .{});
        return false;
    }

    val = -(1 << 8);

    i = 0;
    while (i < len) : (i += 1) val = @max(val, i);

    if (val != test_data[2]) {
        std.log.err("atomicMax failed", .{});
        return false;
    }

    val = 1 << 8;

    i = 0;
    while (i < len) : (i += 1) val = @min(val, i);

    if (val != test_data[3]) {
        std.log.err("atomicMin failed", .{});
        return false;
    }

    var limit: i32 = 17;
    val = 0;

    for (0..@as(usize, @intCast(len)) * loop_num) |_| val = if (val >= limit) 0 else val + 1;

    if (val != test_data[4]) {
        std.log.err("atomicInc failed", .{});
        return false;
    }

    limit = 137;
    val = 0;

    for (0..@as(usize, @intCast(len)) * loop_num) |_| {
        val = if (val == 0 or val > limit) limit else val - 1;
    }

    if (val != test_data[5]) {
        std.log.err("atomicDec failed", .{});
        return false;
    }

    found = false;

    i = 0;
    while (i < len) : (i += 1) {
        if (i == test_data[6]) {
            found = true;
            break;
        }
    }

    if (!found) {
        std.log.err("atomicCAS failed", .{});
        return false;
    }

    val = 0xff;

    i = 0;
    while (i < len) : (i += 1) val &= 2 * i + 7;

    if (val != test_data[7]) {
        std.log.err("atomicAnd failed", .{});
        return false;
    }

    val = 0;

    i = 0;
    while (i < len) : (i += 1) val |= std.math.shl(i32, 1, i);

    if (val != test_data[8]) {
        std.log.err("atomicOr failed", .{});
        return false;
    }

    val = 0xff;

    i = 0;
    while (i < len) : (i += 1) val ^= i;

    if (val != test_data[9]) {
        std.log.err("atomicXor failed", .{});
        return false;
    }

    return true;
}

pub fn main(init: std.process.Init) !void {
    const device = try cuda.init(0);
    defer device.deinit();

    if (try device.attribute(.managed_memory) == 0) {
        std.log.warn("Unified Memory not supported on this device", .{});
        std.process.exit(2);
    }

    const compute_mode_prohibited = 2;
    if (try device.attribute(.compute_mode) == compute_mode_prohibited) {
        std.log.warn("This sample requires a device in either default or " ++
            "process exclusive mode", .{});
        std.process.exit(2);
    }

    const concurrent_managed_access = try device.attribute(.concurrent_managed_access);

    const capability = try device.computeCapability();
    if (capability.major < 6) {
        std.log.warn("systemWideAtomics: requires a minimum CUDA compute 6.0 capability, " ++
            "waiving testing.", .{});
        std.process.exit(2);
    }

    const num_threads = 256;
    const num_blocks = 64;
    const num_data = 10;

    const pageable_memory_access = try device.attribute(.pageable_memory_access) != 0;

    var managed: ?cuda.Buffer(i32) = null;
    defer if (managed) |buffer| buffer.free();
    var atom_arr: []i32 = undefined;
    if (pageable_memory_access) {
        std.log.info("CAN access pageable memory", .{});
        atom_arr = try init.gpa.alloc(i32, num_data);
    } else {
        std.log.info("CANNOT access pageable memory", .{});
        managed = try cuda.allocManaged(i32, num_data);
        atom_arr = managed.?.hostSlice();
    }
    defer if (pageable_memory_access) init.gpa.free(atom_arr);

    @memset(atom_arr, 0);

    atom_arr[7] = 0xff;
    atom_arr[9] = 0xff;

    const module = try cuda.Module.load(@embedFile("systemWideAtomics_kernel.ptx"));
    defer module.unload();
    const atomic_kernel = try module.function("atomicKernel");
    try atomic_kernel.launch(
        .{ .grid = .{ .x = num_blocks }, .block = .{ .x = num_threads } },
        .{@as(u64, @intFromPtr(atom_arr.ptr))},
    );

    if (concurrent_managed_access == 0) try cuda.synchronize();

    atomicKernelCpu(atom_arr, num_blocks * num_threads);

    try cuda.synchronize();

    const test_result = verify(atom_arr, 2 * num_threads * num_blocks);

    const verdict = if (test_result) "OK" else "ERROR!";
    std.log.info("systemWideAtomics completed, returned {s} ", .{verdict});
    if (!test_result) std.process.exit(1);
}
