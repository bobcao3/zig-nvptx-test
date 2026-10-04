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

const sdk_name = "conjugateGradientMultiBlockCG";

const enable_cpu_debug_code = false;
const threads_per_block = 512;

/// Generates a random tridiagonal symmetric matrix.
fn genTridiag(
    random: std.Random,
    i_rows: []i32,
    j_columns: []i32,
    val: []f32,
    n: usize,
    nz: i32,
) void {
    i_rows[0] = 0;
    j_columns[0] = 0;
    j_columns[1] = 1;
    val[0] = random.float(f32) + 10.0;
    val[1] = random.float(f32);

    for (1..n) |i| {
        if (i > 1) {
            i_rows[i] = i_rows[i - 1] + 3;
        } else {
            i_rows[1] = 2;
        }

        const start = (i - 1) * 3 + 2;
        j_columns[start] = @intCast(i - 1);
        j_columns[start + 1] = @intCast(i);

        if (i < n - 1) {
            j_columns[start + 2] = @intCast(i + 1);
        }

        val[start] = val[start - 1];
        val[start + 1] = random.float(f32) + 10.0;

        if (i < n - 1) {
            val[start + 2] = random.float(f32);
        }
    }

    i_rows[n] = nz;
}

/// Computes `outputVecY = alpha * A * inputVecX` for the matrix `A` in CSR format. `i_rows` holds
/// the location of the given non-zero element in the row of the matrix, `j_columns` the location
/// in the column, and `val` the values of the non-zero elements.
fn cpuSpMV(
    i_rows: []const i32,
    j_columns: []const i32,
    val: []const f32,
    num_rows: usize,
    alpha: f32,
    input_vec_x: []const f32,
    output_vec_y: []f32,
) void {
    for (0..num_rows) |i| {
        const num_elems_this_row: usize = @intCast(i_rows[i + 1] - i_rows[i]);
        const row_start: usize = @intCast(i_rows[i]);

        var output: f32 = 0.0;
        for (0..num_elems_this_row) |j| {
            output += alpha * val[row_start + j] * input_vec_x[@intCast(j_columns[row_start + j])];
        }
        output_vec_y[i] = output;
    }
}

fn dotProduct(vec_a: []const f32, vec_b: []const f32) f64 {
    var result: f64 = 0.0;

    for (vec_a, vec_b) |a, b| result = result + (a * b);

    return result;
}

fn scaleVector(vec: []f32, alpha: f32) void {
    for (vec) |*element| element.* = alpha * element.*;
}

fn saxpy(x: []const f32, y: []f32, a: f32) void {
    for (x, y) |x_element, *y_element| y_element.* = a * x_element + y_element.*;
}

fn cpuConjugateGrad(
    i_rows: []const i32,
    j_columns: []const i32,
    val: []const f32,
    x: []f32,
    ax: []f32,
    p: []f32,
    r: []f32,
    n: usize,
    tol: f32,
) void {
    const max_iter = 10000;

    const alpha: f32 = 1.0;
    const alpham1: f32 = -1.0;
    var r0: f32 = 0.0;

    cpuSpMV(i_rows, j_columns, val, n, alpha, x, ax);
    saxpy(ax, r, alpham1);

    var r1: f32 = @floatCast(dotProduct(r, r));

    var k: u32 = 1;

    while (r1 > tol * tol and k <= max_iter) {
        if (k > 1) {
            const b = r1 / r0;
            scaleVector(p, b);

            saxpy(r, p, alpha);
        } else {
            @memcpy(p, r);
        }

        cpuSpMV(i_rows, j_columns, val, n, alpha, p, ax);

        const dot: f32 = @floatCast(dotProduct(p, ax));
        const a = r1 / dot;

        saxpy(p, x, a);
        const na = -a;
        saxpy(ax, r, na);

        r0 = r1;
        r1 = @floatCast(dotProduct(r, r));

        std.log.info("CPU code iteration = {d:>3}, residual = {e}", .{ k, @sqrt(r1) });
        k += 1;
    }
}

pub fn main(init: std.process.Init) !void {
    const tol: f32 = 1e-5;

    std.log.info("Starting [{s}]...", .{sdk_name});

    const device = try cuda.init(0);
    defer device.deinit();

    if (try device.attribute(.managed_memory) == 0) {
        std.log.warn("Unified Memory not supported on this device", .{});
        std.process.exit(2);
    }

    if (try device.attribute(.cooperative_launch) == 0) {
        std.log.warn("Selected GPU (0) does not support Cooperative Kernel Launch, " ++
            "Waiving the run", .{});
        std.process.exit(2);
    }

    const capability = try device.computeCapability();
    const num_sms: u32 = @intCast(try device.attribute(.multiprocessor_count));
    std.log.info("> GPU device has {d} Multi-Processors, SM {d}.{d} compute capabilities", .{
        num_sms,
        capability.major,
        capability.minor,
    });

    const n: usize = 1048576;
    const nz: i32 = (n - 2) * 3 + 4;

    const module = try cuda.Module.load(@embedFile("conjugateGradientMultiBlockCG_kernel.ptx"));
    defer module.unload();
    const kernel = try module.function("gpuConjugateGradient");

    var prng: std.Random.DefaultPrng = .init(1);
    const random = prng.random();

    const i_rows = try cuda.allocManaged(i32, n + 1);
    defer i_rows.free();
    const j_columns = try cuda.allocManaged(i32, @intCast(nz));
    defer j_columns.free();
    const val = try cuda.allocManaged(f32, @intCast(nz));
    defer val.free();

    genTridiag(random, i_rows.hostSlice(), j_columns.hostSlice(), val.hostSlice(), n, nz);

    const x = try cuda.allocManaged(f32, n);
    defer x.free();
    const rhs = try cuda.allocManaged(f32, n);
    defer rhs.free();

    const dot_result = try cuda.allocManaged(f64, 1);
    defer dot_result.free();

    dot_result.hostSlice()[0] = 0.0;

    const r = try cuda.allocManaged(f32, n);
    defer r.free();
    const p = try cuda.allocManaged(f32, n);
    defer p.free();
    const ax = try cuda.allocManaged(f32, n);
    defer ax.free();

    try cuda.synchronize();

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    var ax_cpu: []f32 = undefined;
    var r_cpu: []f32 = undefined;
    var p_cpu: []f32 = undefined;
    var x_cpu: []f32 = undefined;
    if (enable_cpu_debug_code) {
        ax_cpu = try init.gpa.alloc(f32, n);
        r_cpu = try init.gpa.alloc(f32, n);
        p_cpu = try init.gpa.alloc(f32, n);
        x_cpu = try init.gpa.alloc(f32, n);

        for (0..n) |i| {
            r_cpu[i] = 1.0;
            ax_cpu[i] = 0.0;
            x_cpu[i] = 0.0;
        }
    }
    defer if (enable_cpu_debug_code) {
        init.gpa.free(ax_cpu);
        init.gpa.free(r_cpu);
        init.gpa.free(p_cpu);
        init.gpa.free(x_cpu);
    };

    for (r.hostSlice(), rhs.hostSlice(), x.hostSlice()) |*r_element, *rhs_element, *x_element| {
        r_element.* = 1.0;
        rhs_element.* = 1.0;
        x_element.* = 0.0;
    }

    const s_mem_size: u32 = @sizeOf(f64) * ((threads_per_block / 32) + 1);
    const num_threads = threads_per_block;

    const num_blocks_per_sm = try kernel.maxActiveBlocksPerMultiprocessor(num_threads, s_mem_size);

    try start.record(.default);
    try kernel.launchCooperative(.{
        .grid = .{ .x = num_sms * num_blocks_per_sm },
        .block = .{ .x = threads_per_block },
        .shared_memory_bytes = s_mem_size,
    }, .{ i_rows, j_columns, val, x, ax, p, r, dot_result, nz, @as(i32, n), tol });
    try stop.record(.default);
    try cuda.synchronize();

    const time = try start.elapsed(stop);

    const r1: f32 = @floatCast(dot_result.hostSlice()[0]);

    std.log.info(
        "GPU Final, residual = {e}, kernel execution time = {d:.6} ms",
        .{ @sqrt(r1), time },
    );

    if (enable_cpu_debug_code) {
        cpuConjugateGrad(
            i_rows.hostSlice(),
            j_columns.hostSlice(),
            val.hostSlice(),
            x_cpu,
            ax_cpu,
            p_cpu,
            r_cpu,
            n,
            tol,
        );
    }

    var err: f32 = 0.0;

    for (0..n) |i| {
        var rsum: f32 = 0.0;

        const row_start: usize = @intCast(i_rows.hostSlice()[i]);
        const row_end: usize = @intCast(i_rows.hostSlice()[i + 1]);
        for (row_start..row_end) |j| {
            rsum += val.hostSlice()[j] * x.hostSlice()[@intCast(j_columns.hostSlice()[j])];
        }

        const diff = @abs(rsum - rhs.hostSlice()[i]);

        if (diff > err) {
            err = diff;
        }
    }

    std.log.info("Test Summary:  Error amount = {d:.6} ", .{err});
    const passed = @sqrt(r1) < tol;
    std.log.info("&&&& conjugateGradientMultiBlockCG {s}", .{if (passed) "PASSED" else "FAILED"});
    if (!passed) std.process.exit(1);
}
