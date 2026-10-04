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

//! Matrix multiplication: C = A * B. Host code.
//!
//! This sample implements matrix multiplication using the CUDA driver API. It has been written for
//! clarity of exposition to illustrate various CUDA programming principles, not with the goal of
//! providing the most performant generic kernel for matrix multiplication.
//!
//! CUBLAS provides high-performance matrix multiplication. See also:
//! V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
//! in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
//! Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//!
//! Volkov, V. 2010. Better performance at lower occupancy, GPU Technology Conference 2010 (GTC 2010).

const std = @import("std");
const cuda = @import("cuda");

const Selection = struct { function: cuda.Function, block_size: u32 };

fn selectKernel(module: cuda.Module) !Selection {
    const kernels = [_][:0]const u8{ "matrixMul_bs32_64bit", "matrixMul_bs16_64bit", "matrixMul_bs8_64bit" };
    var block_size: u32 = 32;
    for (kernels) |name| {
        const function = try module.function(name);
        const shared_memory_bytes = 2 * block_size * block_size * @sizeOf(f32);
        const potential = try function.maxPotentialBlockSize(shared_memory_bytes, 0);
        if (block_size * block_size <= potential.block_size) {
            std.log.info("> {d} block size selected", .{block_size});
            return .{ .function = function, .block_size = block_size };
        }
        block_size /= 2;
    }
    return .{ .function = try module.function("matrixMul_bs8_64bit"), .block_size = 8 };
}

pub fn main(init: std.process.Init) !void {
    std.log.info("[ matrixMulDrv (Driver API) ]", .{});

    const device = try cuda.init(0);
    defer device.deinit();
    const capability = try device.computeCapability();
    std.log.info("> GPU Device has SM {d}.{d} compute capability", .{ capability.major, capability.minor });
    std.log.info("  Total amount of global memory:     {d} bytes", .{try device.totalMemory()});

    const module = try cuda.Module.load(@embedFile("matrixMul_kernel.ptx"));
    defer module.unload();
    const selection = try selectKernel(module);
    const block_size = selection.block_size;

    const wa = 4 * block_size;
    const ha = 6 * block_size;
    const wb = 4 * block_size;
    const hb = wa;
    const wc = wb;
    const hc = ha;

    const h_a = try init.gpa.alloc(f32, wa * ha);
    defer init.gpa.free(h_a);
    const h_b = try init.gpa.alloc(f32, wb * hb);
    defer init.gpa.free(h_b);
    const val_b: f32 = 0.01;
    @memset(h_a, 1.0);
    @memset(h_b, val_b);

    const d_a = try cuda.dupe(f32, h_a);
    defer d_a.free();
    const d_b = try cuda.dupe(f32, h_b);
    defer d_b.free();
    const d_c = try cuda.alloc(f32, wc * hc);
    defer d_c.free();

    const h_c = try init.gpa.alloc(f32, wc * hc);
    defer init.gpa.free(h_c);

    const start = std.Io.Clock.awake.now(init.io);

    try selection.function.launch(.{
        .grid = .{ .x = wc / block_size, .y = hc / block_size },
        .block = .{ .x = block_size, .y = block_size },
        .shared_memory_bytes = 2 * block_size * block_size * @sizeOf(f32),
    }, .{ d_c, d_a, d_b, @as(usize, wa), @as(usize, wb) });

    try d_c.copyTo(h_c);

    const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io));
    std.log.info("Processing time: {d:.6} (ms)", .{@as(f64, @floatFromInt(elapsed.toNanoseconds())) * 1.0e-6});

    std.log.info("Checking computed result for correctness: ", .{});
    var correct = true;
    const reference = @as(f32, @floatFromInt(wa)) * val_b;
    for (h_c, 0..) |value, i| {
        if (@abs(value - reference) > 1e-5) {
            std.log.err("Error! Matrix[{d:0>5}]={d:.8}, ref={d:.8} error term is > 1e-5", .{ i, value, reference });
            correct = false;
        }
    }

    if (correct) std.log.info("Result = PASS", .{}) else std.log.err("Result = FAIL", .{});
    std.log.info("NOTE: The CUDA Samples are not meant for performance measurements. " ++
        "Results may vary when GPU Boost is enabled.", .{});
    if (!correct) std.process.exit(1);
}
