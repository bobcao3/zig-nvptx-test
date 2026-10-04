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

const sample_name = "[simpleVoteIntrinsics]";

const vote_data_group = 4;

/// Generates the test pattern for Tests 1 and 2.
fn genVoteTestPattern(vote_pattern: []u32) void {
    const size = vote_pattern.len;

    // For testing VOTE.Any (all of these threads will return 0).
    @memset(vote_pattern[0 .. size / 4], 0);

    // For testing VOTE.Any (1/2 these threads will return 1).
    for (2 * size / 8..4 * size / 8) |i| {
        vote_pattern[i] = if (i & 0x01 != 0) @intCast(i) else 0;
    }

    // For testing VOTE.all (1/2 of these threads will return 0).
    for (2 * size / 4..3 * size / 4) |i| {
        vote_pattern[i] = if (i & 0x01 != 0) 0 else @intCast(i);
    }

    // For testing VOTE.all (all of these threads will return 1).
    @memset(vote_pattern[3 * size / 4 .. 4 * size / 4], 0xffffffff);
}

fn printValues(h_result: []const u32, start: usize, end: usize, vote_type: []const u8) void {
    var buffer: [512]u8 = undefined;
    var line: std.Io.Writer = .fixed(&buffer);
    line.print("\t<{s}>[{d} - {d}] = ", .{ vote_type, start, end - 1 }) catch {};
    for (h_result[start..end]) |value| line.print("{d}", .{value}) catch {};
    std.log.err("{s}", .{line.buffered()});
}

fn checkErrors1(h_result: []const u32, start: usize, end: usize, vote_type: []const u8) bool {
    var sum: u32 = 0;
    for (h_result[start..end]) |value| sum += value;

    if (sum > 0) {
        printValues(h_result, start, end, vote_type);
        std.log.err("{d} values FAILED", .{sum});
    }

    return sum > 0;
}

fn checkErrors2(h_result: []const u32, start: usize, end: usize, warp_size: u32, vote_type: []const u8) bool {
    var sum: u32 = 0;
    for (h_result[start..end]) |value| sum += value;

    if (sum != warp_size) {
        printValues(h_result, start, end, vote_type);
        std.log.err(" - FAILED", .{});
    }

    return sum != warp_size;
}

/// Verification code for Kernel #1.
fn checkResultsVoteAnyKernel1(h_result: []const u32, warp_size: u32) u32 {
    var error_count: u32 = 0;
    const quarter = vote_data_group * warp_size / 4;

    error_count += @intFromBool(checkErrors1(h_result, 0, quarter, "Vote.Any"));
    error_count += @intFromBool(checkErrors2(h_result, quarter, 2 * quarter, warp_size, "Vote.Any"));
    error_count += @intFromBool(checkErrors2(h_result, 2 * quarter, 3 * quarter, warp_size, "Vote.Any"));
    error_count += @intFromBool(checkErrors2(h_result, 3 * quarter, 4 * quarter, warp_size, "Vote.Any"));

    if (error_count == 0) std.log.info("\tOK", .{}) else std.log.err("\tERROR", .{});
    return error_count;
}

/// Verification code for Kernel #2.
fn checkResultsVoteAllKernel2(h_result: []const u32, warp_size: u32) u32 {
    var error_count: u32 = 0;
    const quarter = vote_data_group * warp_size / 4;

    error_count += @intFromBool(checkErrors1(h_result, 0, quarter, "Vote.All"));
    error_count += @intFromBool(checkErrors1(h_result, quarter, 2 * quarter, "Vote.All"));
    error_count += @intFromBool(checkErrors1(h_result, 2 * quarter, 3 * quarter, "Vote.All"));
    error_count += @intFromBool(checkErrors2(h_result, 3 * quarter, 4 * quarter, warp_size, "Vote.All"));

    if (error_count == 0) std.log.info("\tOK", .{}) else std.log.err("\tERROR", .{});
    return error_count;
}

/// Verification code for Kernel #3.
fn checkResultsVoteAnyKernel3(hinfo: []const bool, size: usize) u32 {
    var error_count: u32 = 0;

    for (hinfo[0 .. size * 3], 0..) |info, i| {
        const expected = switch (i % 3) {
            // First warp should be all zeros.
            0 => i >= size * 1,
            // First warp and half of second should be all zeros.
            1 => i >= size * 3 / 2,
            // First two warps should be all zeros.
            else => i >= size * 2,
        };
        if (info != expected) error_count += 1;
    }

    if (error_count == 0) std.log.info("\tOK", .{}) else std.log.err("\tERROR", .{});
    return error_count;
}

pub fn main(init: std.process.Init) !void {
    const warp_size = 32;
    var error_count: [3]u32 = .{ 0, 0, 0 };

    std.log.info("{s}", .{sample_name});

    const device = try cuda.init(0);
    defer device.deinit();
    const capability = try device.computeCapability();
    std.log.info(
        "> GPU device has {d} Multi-Processors, SM {d}.{d} compute capabilities",
        .{ try device.attribute(.multiprocessor_count), capability.major, capability.minor },
    );

    const module = try cuda.Module.load(@embedFile("simpleVote_kernel.ptx"));
    defer module.unload();
    const vote_any_kernel1 = try module.function("voteAnyKernel1");
    const vote_all_kernel2 = try module.function("voteAllKernel2");
    const vote_any_kernel3 = try module.function("voteAnyKernel3");

    const h_input = try init.gpa.alloc(u32, vote_data_group * warp_size);
    defer init.gpa.free(h_input);
    const h_result = try init.gpa.alloc(u32, vote_data_group * warp_size);
    defer init.gpa.free(h_result);
    const d_input = try cuda.alloc(u32, h_input.len);
    defer d_input.free();
    const d_result = try cuda.alloc(u32, h_result.len);
    defer d_result.free();
    genVoteTestPattern(h_input);
    try d_input.copyFrom(h_input);

    // Start of Vote Any Test Kernel #1.
    std.log.info("[VOTE Kernel Test 1/3]", .{});
    std.log.info("\tRunning <<Vote.Any>> kernel1 ...", .{});
    {
        try cuda.synchronize();
        try vote_any_kernel1.launch(
            .{ .grid = .{}, .block = .{ .x = vote_data_group * warp_size } },
            .{ d_input, d_result, @as(i32, vote_data_group * warp_size) },
        );
        try cuda.synchronize();
    }
    try d_result.copyTo(h_result);
    error_count[0] += checkResultsVoteAnyKernel1(h_result, warp_size);

    // Start of Vote All Test Kernel #2.
    std.log.info("[VOTE Kernel Test 2/3]", .{});
    std.log.info("\tRunning <<Vote.All>> kernel2 ...", .{});
    {
        try cuda.synchronize();
        try vote_all_kernel2.launch(
            .{ .grid = .{}, .block = .{ .x = vote_data_group * warp_size } },
            .{ d_input, d_result, @as(i32, vote_data_group * warp_size) },
        );
        try cuda.synchronize();
    }
    try d_result.copyTo(h_result);
    error_count[1] += checkResultsVoteAllKernel2(h_result, warp_size);

    // Second Vote Kernel Test #3 (both Any/All).
    const hinfo = try init.gpa.alloc(bool, warp_size * 3 * 3);
    defer init.gpa.free(hinfo);
    @memset(hinfo, false);
    const dinfo = try cuda.dupe(bool, hinfo);
    defer dinfo.free();

    std.log.info("[VOTE Kernel Test 3/3]", .{});
    std.log.info("\tRunning <<Vote.Any>> kernel3 ...", .{});
    {
        try cuda.synchronize();
        try vote_any_kernel3.launch(
            .{ .grid = .{}, .block = .{ .x = warp_size * 3 } },
            .{ dinfo, @as(i32, warp_size) },
        );
        try cuda.synchronize();
    }

    try dinfo.copyTo(hinfo);

    error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

    std.log.info("\tShutting down...", .{});

    if (error_count[0] != 0 or error_count[1] != 0 or error_count[2] != 0) std.process.exit(1);
}
