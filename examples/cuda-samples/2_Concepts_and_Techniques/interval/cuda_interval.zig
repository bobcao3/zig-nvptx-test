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
const lib = @import("cuda_interval_lib.zig");
const RoundedArith = @import("cuda_interval_rounded_arith.zig").RoundedArith;
const constants = @import("interval_constants.zig");

/// Stack in local memory. Managed independently for each work item.
pub fn LocalStack(comptime T: type, comptime n: usize) type {
    return struct {
        buf: [n]T = undefined,
        tos: i32 = -1,

        const Self = @This();

        pub inline fn top(stack: *Self) *T {
            return &stack.buf[@intCast(stack.tos)];
        }

        pub inline fn push(stack: *Self, v: T) void {
            stack.tos += 1;
            stack.buf[@intCast(stack.tos)] = v;
        }

        pub inline fn pop(stack: *Self) T {
            const v = stack.buf[@intCast(stack.tos)];
            stack.tos -= 1;
            return v;
        }

        pub inline fn full(stack: *Self) bool {
            return stack.tos == n - 1;
        }

        pub inline fn empty(stack: *Self) bool {
            return stack.tos == -1;
        }
    };
}

/// Stacks in global memory. Same function as `LocalStack`, but accessible from the host.
/// Interleaved between work items by blocks of `threads` elements. Independent stack for each
/// work item, no sharing of data between work items.
pub fn GlobalStack(comptime T: type, comptime n: i32, comptime threads: i32) type {
    return struct {
        buf: [*]addrspace(.global) T,
        free_index: i32,

        const Self = @This();

        /// `buf` should point to an allocated global buffer of size `n * threads` elements.
        pub inline fn init(buf: [*]addrspace(.global) T, thread_id: i32) Self {
            return .{ .buf = buf, .free_index = thread_id };
        }

        pub inline fn push(stack: *Self, v: T) void {
            stack.buf[@intCast(stack.free_index)] = v;
            stack.free_index += threads;
        }

        pub inline fn pop(stack: *Self) T {
            stack.free_index -= threads;
            return stack.buf[@intCast(stack.free_index)];
        }

        pub inline fn full(stack: *Self) bool {
            return stack.free_index >= n * threads;
        }

        pub inline fn empty(stack: *Self) bool {
            return stack.free_index < threads;
        }

        pub inline fn size(stack: *Self) i32 {
            return @divTrunc(stack.free_index, threads);
        }
    };
}

pub fn Newton(comptime T: type, comptime threads: i32, comptime depth_result: i32) type {
    return struct {
        const I = lib.Interval(T, RoundedArith(T));
        const Result = GlobalStack(I, depth_result, threads);

        /// The function F of which we want to find roots, defined on intervals. Should typically
        /// depend on `thread_id` (indexing an array of coefficients...).
        inline fn f(x: I, thread_id: i32) I {
            const alpha = -@as(T, @floatFromInt(thread_id)) / @as(T, @floatFromInt(threads));
            return x.sub(.point(1)).square().add(I.point(alpha).mul(x));
        }

        /// First derivative of F, also defined on intervals.
        inline fn fd(x: I, thread_id: i32) I {
            const alpha = -@as(T, @floatFromInt(thread_id)) / @as(T, @floatFromInt(threads));
            return I.point(2).mul(x).add(.point(alpha - 2));
        }

        /// Is this interval small enough to stop iterating?
        inline fn isMinimal(x: I, thread_id: i32) bool {
            const epsilon_x: T = 1e-6;
            const epsilon_y: T = 1e-6;
            return !x.isEmpty() and
                (x.width() <= epsilon_x * @abs(x.median()) or f(x, thread_id).width() <= epsilon_y);
        }

        /// In some cases, Newton iterations converge slowly. Bisecting the interval accelerates
        /// convergence.
        inline fn shouldBisect(x: I, x1: I, x2: I, alpha: T) bool {
            const wmax = alpha * x.width();
            return (!x1.isEmpty() and x1.width() > wmax) or (!x2.isEmpty() and x2.width() > wmax);
        }

        /// Main interval Newton loop. Keep refining a list of intervals stored in a stack. Always
        /// keep the next interval to work on in registers (avoids excessive spilling to local mem).
        inline fn newtonInterval(result: *Result, ix0: I, thread_id: i32) void {
            const depth_work = 128;

            const alpha: T = 0.99; // Threshold before switching to bisection

            var work: LocalStack(I, depth_work) = .{};

            var ix = ix0;

            while (true) {
                const x = ix.median();
                const iq = f(.point(x), thread_id);
                const id = fd(ix, thread_id);

                var part2: I = .empty();
                const division = I.divisionPart1(iq, id);
                var has_part2 = division.has_part2;
                var part1 = I.point(x).sub(division.part1).intersect(ix);

                if (has_part2) {
                    part2 = I.point(x).sub(I.divisionPart2(iq, id)).intersect(ix);
                }

                if (isMinimal(part1, thread_id)) {
                    result.push(part1);
                    part1 = .empty();
                }

                if (has_part2 and isMinimal(part2, thread_id)) {
                    result.push(part2);
                    part2 = .empty();
                }

                if (shouldBisect(ix, part1, part2, alpha)) {
                    part1 = .init(ix.lower, x);
                    part2 = .init(x, ix.upper);
                    has_part2 = true;
                }

                if (!part1.isEmpty()) {
                    ix = part1;

                    if (has_part2 and !part2.isEmpty()) {
                        work.push(part2);
                    }
                } else if (has_part2 and !part2.isEmpty()) {
                    ix = part2;
                } else {
                    if (work.empty()) { // If not, we are done
                        break;
                    } else {
                        ix = work.pop(); // Otherwise, pick an interval to work on
                    }
                }
            }
        }

        /// Recursive implementation.
        fn newtonIntervalRec(result: *Result, ix: I, thread_id: i32) void {
            const alpha: T = 0.99; // Threshold before switching to bisection

            if (isMinimal(ix, thread_id)) {
                result.push(ix);
                return;
            }

            const x = ix.median();
            const iq = f(.point(x), thread_id);
            const id = fd(ix, thread_id);

            var part2: I = .empty();
            const division = I.divisionPart1(iq, id);
            var has_part2 = division.has_part2;
            var part1 = I.point(x).sub(division.part1).intersect(ix);

            if (has_part2) {
                part2 = I.point(x).sub(I.divisionPart2(iq, id)).intersect(ix);
            }

            if (shouldBisect(ix, part1, part2, alpha)) {
                part1 = .init(ix.lower, x);
                part2 = .init(x, ix.upper);
                has_part2 = true;
            }

            if (has_part2 and !part2.isEmpty()) {
                newtonIntervalRec(result, part2, thread_id);
            }

            if (!part1.isEmpty()) {
                newtonIntervalRec(result, part1, thread_id);
            }
        }

        /// Naive implementation, no attempt to keep the top of the stack in registers.
        inline fn newtonIntervalNaive(result: *Result, ix0: I, thread_id: i32) void {
            const depth_work = 128;
            const alpha: T = 0.99; // Threshold before switching to bisection

            var work: LocalStack(I, depth_work) = .{};

            work.push(ix0);

            while (!work.empty()) {
                const ix = work.pop();

                if (isMinimal(ix, thread_id)) {
                    result.push(ix);
                } else {
                    const x = ix.median();
                    const iq = f(.point(x), thread_id);
                    const id = fd(ix, thread_id);

                    var part2: I = .empty();
                    const division = I.divisionPart1(iq, id);
                    var has_part2 = division.has_part2;
                    var part1 = I.point(x).sub(division.part1).intersect(ix);

                    if (has_part2) {
                        part2 = I.point(x).sub(I.divisionPart2(iq, id)).intersect(ix);
                    }

                    if (shouldBisect(ix, part1, part2, alpha)) {
                        part1 = .init(ix.lower, x);
                        part2 = .init(x, ix.upper);
                        has_part2 = true;
                    }

                    if (!part1.isEmpty()) {
                        work.push(part1);
                    }

                    if (has_part2 and !part2.isEmpty()) {
                        work.push(part2);
                    }
                }
            }
        }

        pub fn testIntervalNewton(
            buffer: [*]addrspace(.global) I,
            nresults: [*]addrspace(.global) i32,
            i: I,
            implementation_choice: i32,
        ) callconv(.kernel) void {
            const thread_id: i32 = @intCast(@workGroupId(0) * constants.block_size + @workItemId(0));

            var result: Result = .init(buffer, thread_id);

            switch (implementation_choice) {
                0 => newtonIntervalNaive(&result, i, thread_id),
                1 => newtonInterval(&result, i, thread_id),
                2 => newtonIntervalRec(&result, i, thread_id),
                else => newtonIntervalNaive(&result, i, thread_id),
            }

            nresults[@intCast(thread_id)] = result.size();
        }
    };
}
