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

//! Simple CPU implementation. Boost.Interval is not available, so this file computes the
//! directed rounding of the interval bounds from the error of each operation.

const std = @import("std");
const lib = @import("cuda_interval_lib.zig");
const constants = @import("interval_constants.zig");

const T = constants.T;
const threads = constants.threads;
const depth_result = constants.depth_result;
const depth_work = 128;
const use_recursion_cpu = true;

pub fn RoundedMath(comptime Float: type) type {
    return struct {
        fn down(value: Float, error_sign: Float) Float {
            return if (error_sign < 0) std.math.nextAfter(Float, value, -std.math.inf(Float)) else value;
        }

        fn up(value: Float, error_sign: Float) Float {
            return if (error_sign > 0) std.math.nextAfter(Float, value, std.math.inf(Float)) else value;
        }

        fn addError(x: Float, y: Float, sum: Float) Float {
            const virtual_y = sum - x;
            return (x - (sum - virtual_y)) + (y - virtual_y);
        }

        fn divError(x: Float, y: Float, quotient: Float) Float {
            const remainder = @mulAdd(Float, -quotient, y, x);
            return if (y < 0) -remainder else remainder;
        }

        pub fn addDown(x: Float, y: Float) Float {
            const sum = x + y;
            return down(sum, addError(x, y, sum));
        }

        pub fn addUp(x: Float, y: Float) Float {
            const sum = x + y;
            return up(sum, addError(x, y, sum));
        }

        pub fn subDown(x: Float, y: Float) Float {
            return addDown(x, -y);
        }

        pub fn subUp(x: Float, y: Float) Float {
            return addUp(x, -y);
        }

        pub fn mulDown(x: Float, y: Float) Float {
            const product = x * y;
            return down(product, @mulAdd(Float, x, y, -product));
        }

        pub fn mulUp(x: Float, y: Float) Float {
            const product = x * y;
            return up(product, @mulAdd(Float, x, y, -product));
        }

        pub fn divDown(x: Float, y: Float) Float {
            const quotient = x / y;
            return down(quotient, divError(x, y, quotient));
        }

        pub fn divUp(x: Float, y: Float) Float {
            const quotient = x / y;
            return up(quotient, divError(x, y, quotient));
        }

        pub fn median(x: Float, y: Float) Float {
            return (x + y) * 0.5;
        }

        pub fn negInf() Float {
            return -std.math.inf(Float);
        }

        pub fn posInf() Float {
            return std.math.inf(Float);
        }

        pub fn nan() Float {
            return std.math.nan(Float);
        }

        pub fn min(x: Float, y: Float) Float {
            return @min(x, y);
        }

        pub fn max(x: Float, y: Float) Float {
            return @max(x, y);
        }
    };
}

pub const I = lib.Interval(T, RoundedMath(T));

pub fn GlobalStackCpu(comptime Element: type, comptime n: usize, comptime stride: usize) type {
    return struct {
        buf: []Element,
        free_index: usize,

        const Self = @This();

        /// `buf` should point to an allocated global buffer of size `n * stride` elements.
        pub fn init(buf: []Element, thread_id: usize) Self {
            return .{ .buf = buf, .free_index = thread_id };
        }

        pub fn push(stack: *Self, v: Element) void {
            stack.buf[stack.free_index] = v;
            stack.free_index += stride;
        }

        pub fn pop(stack: *Self) Element {
            stack.free_index -= stride;
            return stack.buf[stack.free_index];
        }

        pub fn full(stack: *Self) bool {
            return stack.free_index >= n * stride;
        }

        pub fn empty(stack: *Self) bool {
            return stack.free_index < stride;
        }

        pub fn size(stack: *Self) usize {
            return stack.free_index / stride;
        }
    };
}

const Result = GlobalStackCpu(I, depth_result, threads);

fn fCpu(x: I, thread_id: usize) I {
    const alpha = -@as(T, @floatFromInt(thread_id)) / @as(T, @floatFromInt(threads));
    return x.sub(.point(1)).square().add(I.point(alpha).mul(x));
}

fn fdCpu(x: I, thread_id: usize) I {
    const alpha = -@as(T, @floatFromInt(thread_id)) / @as(T, @floatFromInt(threads));
    return I.point(2).mul(x).add(.point(alpha - 2));
}

fn isMinimalCpu(x: I, thread_id: usize) bool {
    const epsilon_x: T = 1e-6;
    const epsilon_y: T = 1e-6;
    return !x.isEmpty() and
        (x.width() <= epsilon_x * @abs(x.median()) or fCpu(x, thread_id).width() <= epsilon_y);
}

fn shouldBisectCpu(x: I, x1: I, x2: I, alpha: T) bool {
    const wmax = alpha * x.width();
    return x1.width() > wmax or x2.width() > wmax;
}

fn newtonIntervalCpu(result: *Result, ix0: I, thread_id: usize) void {
    const alpha: T = 0.99;

    var local_buffer: [depth_work]I = undefined;
    var work: GlobalStackCpu(I, depth_work, 1) = .init(&local_buffer, 0);

    var ix = ix0;

    while (true) {
        const x = ix.median();
        const iq = fCpu(.point(x), thread_id);
        const id = fdCpu(ix, thread_id);

        const division = I.divisionPart1(iq, id);
        var has_part2 = division.has_part2;
        var part1 = I.point(x).sub(division.part1).intersect(ix);
        var part2: I = undefined;

        if (has_part2) {
            part2 = I.point(x).sub(I.divisionPart2(iq, id)).intersect(ix);
        }

        if (isMinimalCpu(part1, thread_id)) {
            result.push(part1);
            part1 = .empty();
        }

        if (has_part2 and isMinimalCpu(part2, thread_id)) {
            result.push(part2);
            part2 = .empty();
        }

        if (shouldBisectCpu(ix, part1, part2, alpha)) {
            part1 = .init(ix.lower, x);
            part2 = .init(x, ix.upper);
            has_part2 = true;
        }

        if (part1.lower <= part1.upper and !part1.isEmpty()) {
            ix = part1;

            if (has_part2 and !part2.isEmpty()) {
                work.push(part2);
            }
        } else if (has_part2 and !part2.isEmpty()) {
            ix = part2;
        } else {
            if (work.empty()) {
                break;
            } else {
                ix = work.pop();
            }
        }
    }
}

fn newtonIntervalRecCpu(result: *Result, ix: I, thread_id: usize) void {
    const alpha: T = 0.99;

    if (isMinimalCpu(ix, thread_id)) {
        result.push(ix);
        return;
    }

    const x = ix.median();
    const iq = fCpu(.point(x), thread_id);
    const id = fdCpu(ix, thread_id);

    const division = I.divisionPart1(iq, id);
    var has_part2 = division.has_part2;
    var part1 = I.point(x).sub(division.part1).intersect(ix);
    var part2: I = undefined;

    if (has_part2) {
        part2 = I.point(x).sub(I.divisionPart2(iq, id)).intersect(ix);
    }

    if (shouldBisectCpu(ix, part1, part2, alpha)) {
        part1 = .init(ix.lower, x);
        part2 = .init(x, ix.upper);
        has_part2 = true;
    }

    if (part1.lower <= part1.upper and !part1.isEmpty()) {
        newtonIntervalRecCpu(result, part1, thread_id);
    }

    if (has_part2 and !part2.isEmpty()) {
        newtonIntervalRecCpu(result, part2, thread_id);
    }
}

pub fn testIntervalNewtonCpu(buffer: []I, nresults: []i32, i: I) void {
    for (0..threads) |thread_id| {
        var result: Result = .init(buffer, thread_id);

        if (use_recursion_cpu) {
            newtonIntervalRecCpu(&result, i, thread_id);
        } else {
            newtonIntervalCpu(&result, i, thread_id);
        }
        nresults[thread_id] = @intCast(result.size());
    }
}

pub fn checkAgainstHost(
    h_nresults: []const i32,
    h_nresults_cpu: []const i32,
    h_result: []const I,
    h_result_cpu: []const I,
) bool {
    std.log.info("Check against Host computation...", .{});
    var success1 = true;
    var success2 = true;

    if (h_nresults_cpu[0] != h_nresults[0]) return false;

    const count: usize = @intCast(h_nresults[0]);
    for (0..count) |i| {
        const diff1 = @abs(h_result[threads * i].lower - h_result_cpu[threads * i].lower);
        const diff2 = @abs(h_result[threads * i].upper - h_result_cpu[threads * i].upper);

        if (diff1 > 1.0e-6 or diff2 > 1.0e-6) {
            success1 = false;
            break;
        }
    }

    // In case the two intervals are reversed.
    for (0..count) |i| {
        const reversed = threads * (count - i - 1);
        const diff1 = @abs(h_result[threads * i].lower - h_result_cpu[reversed].lower);
        const diff2 = @abs(h_result[threads * i].upper - h_result_cpu[reversed].upper);

        if (diff1 > 1.0e-6 or diff2 > 1.0e-6) {
            success2 = false;
            break;
        }
    }

    return success1 or success2;
}
