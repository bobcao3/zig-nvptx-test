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

//! This sample implements a Niederreiter quasirandom number generator and the Moro inverse
//! cumulative normal distribution function, and compares both with a CPU implementation.

const std = @import("std");
const cuda = @import("cuda");
const common = @import("quasirandomGenerator_common.zig");

const qrng_dimensions = common.qrng_dimensions;
const qrng_resolution = common.qrng_resolution;

const Table = [qrng_dimensions][qrng_resolution]u32;

/// The internal 63-bit table of the generator.
var cjn: [63][qrng_dimensions]i64 = undefined;

fn bit(exponent: i32) i32 {
    return @as(i32, 1) << @intCast(exponent);
}

fn shiftLeft(value: i32, amount: i32) i32 {
    return value << @intCast(amount);
}

fn generatePolynomials(buffer: *[qrng_dimensions]i32, primitive: bool) i32 {
    var p2: i32 = 0;
    var l: i32 = 0;
    var e_p2: i32 = 0;

    buffer[0] = 0x2;
    for (1..qrng_dimensions) |count| {
        // Searches for the next irreducible polynomial.
        var p1 = buffer[count - 1] + 1;
        while (true) : (p1 += 1) {
            // Finds the degree of the polynomial p1.
            var e_p1: i32 = 30;
            while (p1 & bit(e_p1) == 0) e_p1 -= 1;

            // Tries to divide p1 by all polynomials in the buffer.
            for (buffer[0..count]) |divisor| {
                // Finds the degree of the divisor.
                var e_b = e_p1;
                while (divisor & bit(e_b) == 0) e_b -= 1;

                // Divides p2 by the divisor until the end.
                e_p2 = e_p1;
                p2 = shiftLeft(divisor, e_p2 - e_b) ^ p1;
                while (p2 >= divisor) : (p2 = shiftLeft(divisor, e_p2 - e_b) ^ p2) {
                    // Computes the new degree of p2.
                    while (p2 & bit(e_p2) == 0) e_p2 -= 1;
                }

                // A division without remainder means that p1 is not irreducible.
                if (p2 == 0) break;
            }

            // All divisions had a remainder, so p1 is irreducible.
            if (p2 != 0) {
                e_p2 = 0;

                if (primitive) {
                    // Checks that p1 has only one cycle, that is, p1 is monic or primitive.
                    const j: i32 = @bitCast(~(@as(u32, 0xffffffff) << @intCast(e_p1 + 1)));
                    const e_b = bit(e_p1) | 0x1;

                    p2 = e_b;
                    e_p2 = bit(e_p1) - 2;
                    while (e_p2 > 0) : (e_p2 -= 1) {
                        p2 <<= 1;
                        var i = p2 & p1;
                        i = (i & 0x55555555) + ((i >> 1) & 0x55555555);
                        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
                        i = (i & 0x07070707) + ((i >> 4) & 0x07070707);
                        p2 |= @rem(i, 255) & 1;

                        if ((p2 & j) == e_b) break;
                    }
                }

                // The polynomial is monic, so it joins the list of polynomials.
                if (e_p2 == 0) {
                    buffer[count] = p1;
                    l += e_p1;
                    break;
                }
            }
        }
    }

    return l + 1;
}

fn polynomialPower(p: []const i32) usize {
    var e: usize = 0;
    while (p[e + 1] != -1) e += 1;
    return e;
}

/// Generates the table of the Niederreiter sequence. B. Fox, P. Bratley and H. Niederreiter,
/// "Implementation and test of low discrepancy sequences", ACM Trans. Model. Comput. Simul.,
/// 2(3):195-213, July 1992.
fn generateCj(gpa: std.mem.Allocator) !void {
    var buffer: [qrng_dimensions]i32 = undefined;

    // Niederreiter, in contrast to Sobol, allows irreducible polynomials that are not primitive.
    const l: usize = @intCast(generatePolynomials(&buffer, false));

    // Converts all polynomials from the buffer to the polynomials table.
    const polynomials = try gpa.alloc(i32, l + 2 * qrng_dimensions + 1);
    defer gpa.free(polynomials);

    var next: usize = 0;
    for (buffer) |p1| {
        // Finds the degree of the polynomial p1.
        var e_p1: i32 = 30;
        while (p1 & bit(e_p1) == 0) e_p1 -= 1;

        // Fills the polynomials table with the values of this polynomial.
        polynomials[next] = 1;
        next += 1;

        e_p1 -= 1;
        while (e_p1 >= 0) : (e_p1 -= 1) {
            polynomials[next] = (p1 >> @intCast(e_p1)) & 1;
            next += 1;
        }

        polynomials[next] = -1;
        next += 1;
    }

    polynomials[next] = -1;

    // The irreducible polynomial p.
    var p: []const i32 = polynomials;
    // The polynomial b, the array v and the temporary polynomial t, which holds b during the
    // multiplication of p and b.
    var b_arr: [1024]i32 = @splat(0);
    var v_arr: [1024]i32 = @splat(0);
    var t_arr: [1024]i32 = @splat(0);

    // Cycles over the monic irreducible polynomials.
    var d: usize = 0;
    while (p[0] != -1) : (d += 1) {
        // Allocates the memory for the cj array for the dimension d.
        for (&cjn) |*row| row[d] = 0;

        // Determines the power of the irreducible polynomial.
        const e = polynomialPower(p);

        // The polynomial b in the beginning is just 1.
        var m: usize = 0;
        var b: []i32 = b_arr[1023..];
        b[m] = 1;
        // The array v needs only a length of 63 + e - 2.
        const v = v_arr[1023 - (63 + e - 2) ..];
        var t: []i32 = undefined;

        // Cycles over all coefficients.
        var u = e;
        var j: i32 = 63 - 1;
        while (j >= 0) : ({
            j -= 1;
            u += 1;
        }) {
            if (u == e) {
                u = 0;

                // Multiplies b by p, which is a multiplication of polynomials.
                const m1 = m;
                t = t_arr[1023 - m1 ..];
                for (0..m + 1) |i| t[i] = b[i];

                m += e;
                b = b_arr[1023 - m ..];

                for (0..m + 1) |i| {
                    b[i] = 0;

                    var ip: i32 = @as(i32, @intCast(e)) - @as(i32, @intCast(m - i));
                    var it: i32 = @intCast(m1);
                    while (ip <= @as(i32, @intCast(e)) and it >= 0) : ({
                        ip += 1;
                        it -= 1;
                    }) {
                        if (ip >= 0) b[i] ^= p[@intCast(ip)] & t[@intCast(it)];
                    }
                }

                // Calculates v.
                var i: usize = 0;
                while (i < m1) : (i += 1) v[i] = 0;
                while (i < m) : (i += 1) v[i] = 1;
                while (i <= 63 + e - 2) : (i += 1) {
                    v[i] = 0;

                    for (1..m + 1) |it| v[i] ^= v[i - it] & b[it];
                }
            }

            // Copies the calculated v to cj.
            for (0..63) |i| {
                cjn[i][d] |= @as(i64, v[i + u]) << @intCast(j);
            }
        }

        p = p[e + 2 ..];
    }
}

/// Generates a 63-bit quasirandom number for the index and the dimension and normalizes it.
fn getQuasirandomValue63(index: i64, dim: usize) f64 {
    const int63_scale = 1.0 / @as(f64, @floatFromInt(@as(u64, 0x8000000000000001)));
    var result: i64 = 0;

    var i = index;
    for (0..63) |bit_index| {
        if (i & 1 != 0) result ^= cjn[bit_index][dim];
        i >>= 1;
    }

    return @as(f64, @floatFromInt(result + 1)) * int63_scale;
}

fn initQuasirandomGenerator(gpa: std.mem.Allocator, table: *Table) !void {
    try generateCj(gpa);

    for (table, 0..) |*row, dim| {
        for (row, 0..) |*entry, bit_index| {
            entry.* = @intCast((cjn[bit_index][dim] >> 32) & 0x7FFFFFFF);
        }
    }
}

/// Moro's approximation of the inverse cumulative normal distribution function.
fn moroInvCndCpu(input: u32) f64 {
    const a1 = 2.50662823884;
    const a2 = -18.61500062529;
    const a3 = 41.39119773534;
    const a4 = -25.44106049637;
    const b1 = -8.4735109309;
    const b2 = 23.08336743743;
    const b3 = -21.06224101826;
    const b4 = 3.13082909833;
    const c1 = 0.337475482272615;
    const c2 = 0.976169019091719;
    const c3 = 0.160797971491821;
    const c4 = 2.76438810333863E-02;
    const c5 = 3.8405729373609E-03;
    const c6 = 3.951896511919E-04;
    const c7 = 3.21767881768E-05;
    const c8 = 2.888167364E-07;
    const c9 = 3.960315187E-07;

    var x = input;
    var z: f64 = undefined;
    var negate = false;

    // Restricts the input to the bottom half of the input domain, so that the conversion to
    // floating point gives a value in the range (0,0.5]. The result is reflected later if the
    // input was in the top half of the domain.
    if (x >= 0x80000000) {
        x = 0xffffffff - x;
        negate = true;
    }

    // x is now in the range [0,0x7fffffff]. Converts it to floating point in (0,0.5].
    const x1 = 1.0 / @as(f64, @floatFromInt(@as(u32, 0xffffffff)));
    const x2 = x1 / 2.0;
    const p1 = @as(f64, @floatFromInt(x)) * x1 + x2;
    // Converts to floating point in (-0.5,0].
    const p2 = p1 - 0.5;

    // Main body of the bell curve for |p| < 0.42.
    if (p2 > -0.42) {
        z = p2 * p2;
        const numerator = ((a4 * z + a3) * z + a2) * z + a1;
        const denominator = (((b4 * z + b3) * z + b2) * z + b1) * z + 1.0;
        z = p2 * numerator / denominator;
    } else {
        // Special case (Chebychev) for the tail.
        z = @log(-@log(p1));
        const upper = c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)));
        z = -(c1 + z * (c2 + z * (c3 + z * (c4 + z * upper))));
    }

    return if (negate) -z else z;
}

const n = 1048576;

const Gpu = struct {
    quasirandom_generator_kernel: cuda.Function,
    inverse_cnd_kernel: cuda.Function,

    fn init(module: cuda.Module, table_cpu: *const Table) !Gpu {
        const c_table = try module.global(Table, "c_table");
        try c_table.copyFrom(@as([*]const Table, @ptrCast(table_cpu))[0..1]);
        return .{
            .quasirandom_generator_kernel = try module.function("quasirandomGeneratorKernel"),
            .inverse_cnd_kernel = try module.function("inverseCndKernel"),
        };
    }

    fn quasirandomGenerator(gpu: Gpu, d_output: cuda.Buffer(f32), seed: u32, count: u32) !void {
        try gpu.quasirandom_generator_kernel.launch(.{
            .grid = .{ .x = 128 },
            .block = .{ .x = 128, .y = qrng_dimensions },
        }, .{ d_output, seed, count });
    }

    /// Transforms the numbers of `d_input`, or uniformly placed samples when `d_input` is null.
    fn inverseCnd(gpu: Gpu, d_output: cuda.Buffer(f32), d_input: ?cuda.Buffer(u32), count: u32) !void {
        try gpu.inverse_cnd_kernel.launch(.{
            .grid = .{ .x = 128 },
            .block = .{ .x = 128 },
        }, .{ d_output, d_input, count });
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    std.log.info("quasirandomGenerator Starting...", .{});

    var table_cpu: Table = undefined;

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("quasirandomGenerator_kernel.ptx"));
    defer module.unload();

    std.log.info("Allocating GPU memory...", .{});
    const d_output = try cuda.alloc(f32, qrng_dimensions * n);
    defer d_output.free();

    std.log.info("Allocating CPU memory...", .{});
    const h_output_gpu = try gpa.alloc(f32, qrng_dimensions * n);
    defer gpa.free(h_output_gpu);

    std.log.info("Initializing QRNG tables...", .{});
    try initQuasirandomGenerator(gpa, &table_cpu);

    const gpu: Gpu = try .init(module, &table_cpu);

    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();

    std.log.info("Testing QRNG...", .{});
    try d_output.memset(0);
    const num_iterations = 20;

    // The first iteration is a warmup iteration.
    try gpu.quasirandomGenerator(d_output, 0, n);
    try cuda.synchronize();
    try start.record(.default);
    for (0..num_iterations) |_| try gpu.quasirandomGenerator(d_output, 0, n);
    try stop.record(.default);
    try stop.synchronize();
    var gpu_time: f64 = 1e-3 * try start.elapsed(stop) / num_iterations;
    std.log.info("quasirandomGenerator, Throughput = {d:.4} GNumbers/s, Time = {d:.5} s, Size " ++
        "= {d} Numbers, NumDevsUsed = {d}, Workgroup = {d}", .{
        @as(f64, qrng_dimensions) * @as(f64, n) * 1.0e-9 / gpu_time,
        gpu_time,
        qrng_dimensions * n,
        1,
        128 * qrng_dimensions,
    });

    std.log.info("Reading GPU results...", .{});
    try d_output.copyTo(h_output_gpu);

    std.log.info("Comparing to the CPU results...", .{});
    var sum_delta: f64 = 0;
    var sum_ref: f64 = 0;

    for (0..qrng_dimensions) |dim| {
        for (0..n) |pos| {
            const reference = getQuasirandomValue63(@intCast(pos), dim);
            const delta = @as(f64, h_output_gpu[dim * n + pos]) - reference;
            sum_delta += @abs(delta);
            sum_ref += @abs(reference);
        }
    }

    std.log.info("L1 norm: {E}", .{sum_delta / sum_ref});

    std.log.info("Testing inverseCNDgpu()...", .{});
    try d_output.memset(0);

    try gpu.inverseCnd(d_output, null, qrng_dimensions * n);
    try cuda.synchronize();
    try start.record(.default);
    for (0..num_iterations) |_| try gpu.inverseCnd(d_output, null, qrng_dimensions * n);
    try stop.record(.default);
    try stop.synchronize();
    gpu_time = 1e-3 * try start.elapsed(stop) / num_iterations;
    std.log.info("quasirandomGenerator-inverse, Throughput = {d:.4} GNumbers/s, Time = {d:.5} " ++
        "s, Size = {d} Numbers, NumDevsUsed = {d}, Workgroup = {d}", .{
        @as(f64, qrng_dimensions) * @as(f64, n) * 1e-9 / gpu_time,
        gpu_time,
        qrng_dimensions * n,
        1,
        128,
    });

    std.log.info("Reading GPU results...", .{});
    try d_output.copyTo(h_output_gpu);

    std.log.info("Comparing to the CPU results...", .{});
    sum_delta = 0;
    sum_ref = 0;
    const distance = std.math.maxInt(u32) / (qrng_dimensions * n + 1);

    for (h_output_gpu, 0..) |value, pos| {
        const d: u32 = @intCast((pos + 1) * distance);
        const reference = moroInvCndCpu(d);
        const delta = @as(f64, value) - reference;
        sum_delta += @abs(delta);
        sum_ref += @abs(reference);
    }

    const l1_norm = sum_delta / sum_ref;
    std.log.info("L1 norm: {E}", .{l1_norm});

    std.log.info("Shutting down...", .{});

    if (!(l1_norm < 1e-6)) std.process.exit(1);
}
