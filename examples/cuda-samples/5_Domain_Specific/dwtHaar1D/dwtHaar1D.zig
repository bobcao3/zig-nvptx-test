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

//! 1D DWT for Haar wavelet and signals with a length which is a power of 2.
//! The code reduces bank conflicts and non-coalesced reads / writes as
//! appropriate but does not fully remove them because the computational
//! overhead to achieve this would outweighs the benefit (see inline comments
//! for more details).
//! Large signals are subdivided into sub-signals with 512 elements and the
//! wavelet transform for these is computed with one block over 10 decomposition
//! levels. The resulting signal consisting of the approximation coefficients at
//! level X is then processed in a subsequent step on the device. This requires
//! interblock synchronization which is only possible on host side.
//! Detail coefficients which have been computed are not further referenced
//! during the decomposition so that they can be stored directly in their final
//! position in global memory. The transform and its storing scheme preserve
//! locality in the coefficients so that these writes are coalesced.
//! Approximation coefficients are stored in shared memory because they are
//! needed to compute the subsequent decomposition step. The top most
//! approximation coefficient for a sub-signal processed by one block is stored
//! in a special global memory location to simplify the processing after the
//! interblock synchronization.
//! Most books on wavelets explain the Haar wavelet decomposition. A good freely
//! available resource is the Wavelet primer by Stollnitz et al.
//! http://grail.cs.washington.edu/projects/wavelets/article/wavelet1.pdf
//! http://grail.cs.washington.edu/projects/wavelets/article/wavelet2.pdf
//! The basic of all Wavelet transforms is to decompose a signal into
//! approximation (a) and detail (d) coefficients where the detail tends to be
//! small or zero which allows / simplifies compression. The following "graphs"
//! demonstrate the transform for a signal
//! of length eight. The index always describes the decomposition level where
//! a coefficient arises. The input signal is interpreted as approximation signal
//! at level 0. The coefficients computed on the device are stored in the same
//! scheme as in the example. This data structure is particularly well suited for
//! compression and also preserves the hierarchical structure of the
//! decomposition.
//!
//!     -------------------------------------------------
//!     | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 |
//!     -------------------------------------------------
//!
//!     -------------------------------------------------
//!     | a_1 | a_1 | a_1 | a_1 | d_1 | d_1 | d_1 | d_1 |
//!     -------------------------------------------------
//!
//!     -------------------------------------------------
//!     | a_2 | a_2 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
//!     -------------------------------------------------
//!
//!     -------------------------------------------------
//!     | a_3 | d_3 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
//!     -------------------------------------------------
//!
//! Host code.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("dwtHaar1D_common.zig");

const signal_file = @embedFile("signal.dat");
const gold_file = @embedFile("regression.gold.dat");

const result_file = "result.dat";

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();

    std.log.info("source file    = \"signal.dat\"", .{});
    std.log.info("reference file = \"{s}\"", .{result_file});
    std.log.info("gold file      = \"regression.gold.dat\"", .{});

    const signal = try parseFloats(gpa, signal_file);
    defer gpa.free(signal);
    std.log.info("Reading signal from \"signal.dat\"", .{});

    const dlevels_complete = getLevels(signal.len) orelse {
        std.log.err("Signal length not supported.", .{});
        std.process.exit(1);
    };

    const module = try cuda.Module.load(@embedFile("dwtHaar1D_kernel.ptx"));
    defer module.unload();
    const init_value = try module.function("initValue");
    const dwt_haar1d = try module.function("dwtHaar1D");

    const d_idata = try cuda.dupe(f32, signal);
    defer d_idata.free();
    const d_odata = try cuda.alloc(f32, signal.len);
    defer d_odata.free();
    const approx_final = try cuda.alloc(f32, signal.len);
    defer approx_final.free();

    // The very final approximation coefficient has to be written to the output data, all others
    // are reused as input data in the next global step and therefore have to be written to the
    // input data again.
    var approx_is_input: bool = undefined;

    var num_threads_total_left: u32 = @intCast(signal.len / 2);
    var dlevels_step: u32 = dlevels_complete;

    var block_size: cuda.Dim3 = .{};
    var grid_size: cuda.Dim3 = .{};

    var dlevels_left = dlevels_complete;

    if (dlevels_complete <= 10) {
        block_size.x = num_threads_total_left;
        approx_is_input = false;
    } else {
        grid_size.x = num_threads_total_left / 512;
        block_size.x = 512;

        dlevels_step = 10;
        dlevels_left -= 10;

        approx_is_input = true;
    }

    try init_value.launch(.{ .grid = grid_size, .block = block_size }, .{ d_odata, @as(f32, 0.0) });

    while (num_threads_total_left != 0) {
        var mem_shared = (2 * block_size.x) * @sizeOf(f32);
        mem_shared += ((2 * block_size.x) / common.num_banks) * @sizeOf(f32);

        try dwt_haar1d.launch(.{
            .grid = grid_size,
            .block = block_size,
            .shared_memory_bytes = mem_shared,
        }, .{ d_idata, d_odata, approx_final, dlevels_step, num_threads_total_left, block_size.x });

        const approx_coefficients: cuda.Buffer(f32) = .{
            .address = approx_final.address,
            .len = grid_size.x,
        };
        if (approx_is_input) {
            try d_idata.copyFromBuffer(approx_coefficients);
        } else {
            try d_odata.copyFromBuffer(approx_coefficients);
        }

        if (dlevels_left < 10) {
            approx_is_input = false;
        }

        dlevels_step = if (dlevels_left > 10) dlevels_left - 10 else dlevels_left;
        dlevels_left -%= 10;

        // After each step only half the threads are used any longer, therefore after 10 steps
        // 2^10 less threads.
        num_threads_total_left = num_threads_total_left >> 10;

        grid_size.x = if (num_threads_total_left / 512 + @intFromBool(num_threads_total_left % 512 != 0) != 0) 1 else 0;

        if (grid_size.x <= 1) {
            block_size.x = num_threads_total_left;
        }
    }

    const odata = try gpa.alloc(f32, signal.len);
    defer gpa.free(odata);
    try d_odata.copyTo(odata);

    try writeFloats(gpa, init.io, result_file, odata, 0.001);
    std.log.info("Writing result to \"{s}\"", .{result_file});

    const reference = try parseFloats(gpa, gold_file);
    defer gpa.free(reference);
    std.log.info("Reading reference result from \"regression.gold.dat\"", .{});

    std.debug.assert(signal.len == reference.len);

    const passed = compareL2fe(reference, odata, 0.001);

    if (passed) std.log.info("Test success!", .{}) else std.log.err("Test failure!", .{});
    if (!passed) std.process.exit(1);
}

/// Returns the number of decomposition levels necessary to perform a full decomposition, or null
/// if the length of the signal is not supported (signals up to a length of 2^20 are supported).
fn getLevels(len: usize) ?u32 {
    for (0..20) |i| {
        if (len == @as(usize, 1) << @intCast(i)) return @intCast(i);
    }
    return null;
}

fn parseFloats(gpa: std.mem.Allocator, text: []const u8) ![]f32 {
    var values: std.ArrayList(f32) = .empty;
    errdefer values.deinit(gpa);
    var tokens = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (tokens.next()) |token| try values.append(gpa, try std.fmt.parseFloat(f32, token));
    return values.toOwnedSlice(gpa);
}

fn writeFloats(
    gpa: std.mem.Allocator,
    io: std.Io,
    sub_path: []const u8,
    data: []const f32,
    epsilon: f32,
) !void {
    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    try text.writer.print("# {d}\n", .{epsilon});
    for (data) |value| try text.writer.print("{d} ", .{value});
    try text.writer.writeByte('\n');
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sub_path, .data = text.written() });
}

fn compareL2fe(reference: []const f32, data: []const f32, epsilon: f32) bool {
    var err: f32 = 0;
    var ref: f32 = 0;

    for (reference, data) |expected, actual| {
        const diff = expected - actual;
        err += diff * diff;
        ref += expected * expected;
    }

    const norm_ref = @sqrt(ref);
    if (@abs(ref) < 1e-7) return false;

    const norm_error = @sqrt(err);
    return norm_error / norm_ref < epsilon;
}
