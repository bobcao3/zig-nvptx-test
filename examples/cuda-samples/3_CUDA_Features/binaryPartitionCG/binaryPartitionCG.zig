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

//! This sample illustrates basic usage of binary partition cooperative groups within the thread
//! block tile when divergent path exists.
//! 1. Each thread loads a value from random array.
//! 2. Then it checks if the value is odd or even.
//! 3. It creates a binary partition group based on the above predicate.
//! 4. It counts the number of odd and even values in the group based on the size of the binary
//!    groups.
//! 5. It writes the count to a global counter of odd values.
//! 6. It sums the values loaded by the individual threads (using reduce) and writes the sums to
//!    the global even and odd element sums.
//!
//! **NOTE**:
//! binary_partition results in splitting warp into divergent thread groups. This is not good from
//! performance perspective, but in cases where warp divergence is inevitable one can use
//! binary_partition group.

const std = @import("std");
const cuda = @import("cuda");

fn initOddEvenArr(input_arr: []i32, random: std.Random) void {
    for (input_arr) |*element| element.* = random.intRangeLessThan(i32, 0, 50);
}

pub fn main(init: std.process.Init) !void {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("binaryPartitionCG_kernel.ptx"));
    defer module.unload();
    const odd_even_count_and_sum_cg = try module.function("oddEvenCountAndSumCg");

    const arr_size = 1024 * 100;
    const h_input_arr = try init.gpa.alloc(i32, arr_size);
    defer init.gpa.free(h_input_arr);
    var h_num_of_odds: [1]i32 = undefined;
    var h_sum_of_odd_even_elems: [2]i32 = undefined;
    var prng: std.Random.DefaultPrng = .init(2006);
    initOddEvenArr(h_input_arr, prng.random());

    const stream = try cuda.Stream.create();
    defer stream.destroy();
    const d_input_arr = try cuda.alloc(i32, arr_size);
    defer d_input_arr.free();
    const d_num_of_odds = try cuda.alloc(i32, 1);
    defer d_num_of_odds.free();
    const d_sum_of_odd_even_elems = try cuda.alloc(i32, 2);
    defer d_sum_of_odd_even_elems.free();

    try d_input_arr.copyFromAsync(h_input_arr, stream);
    try d_num_of_odds.memsetAsync(0, stream);
    try d_sum_of_odd_even_elems.memsetAsync(0, stream);

    const potential = try odd_even_count_and_sum_cg.maxPotentialBlockSize(0, 0);
    const blocks_per_grid = potential.min_grid_size;
    const threads_per_block = potential.block_size;

    std.log.info("Launching {d} blocks with {d} threads...", .{ blocks_per_grid, threads_per_block });

    try odd_even_count_and_sum_cg.launch(
        .{ .grid = .{ .x = blocks_per_grid }, .block = .{ .x = threads_per_block }, .stream = stream },
        .{ d_input_arr, d_num_of_odds, d_sum_of_odd_even_elems, @as(u32, arr_size) },
    );

    try d_num_of_odds.copyToAsync(&h_num_of_odds, stream);
    try d_sum_of_odd_even_elems.copyToAsync(&h_sum_of_odd_even_elems, stream);
    try stream.synchronize();

    std.log.info("Array size = {d} Num of Odds = {d} Sum of Odds = {d} Sum of Evens {d}", .{
        arr_size,
        h_num_of_odds[0],
        h_sum_of_odd_even_elems[0],
        h_sum_of_odd_even_elems[1],
    });
    std.log.info("...Done.", .{});
}
