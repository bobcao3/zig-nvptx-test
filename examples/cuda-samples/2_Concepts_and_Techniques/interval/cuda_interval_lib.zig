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

//! Interval type and basic operations. The interface follows the Boost Interval library
//! (www.boost.org). `Arith` supplies the rounded arithmetic.

pub fn Interval(comptime T: type, comptime Arith: type) type {
    return extern struct {
        lower: T,
        upper: T,

        const Self = @This();
        pub const Base = T;

        /// The result of `division`: the first part, and the second part if the divisor contains 0.
        pub const Division = struct { part1: Self, has_part2: bool };

        pub inline fn init(l: T, u: T) Self {
            return .{ .lower = l, .upper = u };
        }

        pub inline fn point(v: T) Self {
            return .{ .lower = v, .upper = v };
        }

        pub inline fn empty() Self {
            return .{ .lower = Arith.nan(), .upper = Arith.nan() };
        }

        pub inline fn isEmpty(x: Self) bool {
            const hash = x.lower + x.upper;
            return hash != hash;
        }

        pub inline fn width(x: Self) T {
            if (x.isEmpty()) return 0;

            return Arith.subUp(x.upper, x.lower);
        }

        pub inline fn neg(x: Self) Self {
            return init(-x.upper, -x.lower);
        }

        pub inline fn add(x: Self, y: Self) Self {
            return init(Arith.addDown(x.lower, y.lower), Arith.addUp(x.upper, y.upper));
        }

        pub inline fn sub(x: Self, y: Self) Self {
            return init(Arith.subDown(x.lower, y.upper), Arith.subUp(x.upper, y.lower));
        }

        pub inline fn mul(x: Self, y: Self) Self {
            // Textbook implementation: 14 flops, but no branch.
            return init(
                min4(
                    Arith.mulDown(x.lower, y.lower),
                    Arith.mulDown(x.lower, y.upper),
                    Arith.mulDown(x.upper, y.lower),
                    Arith.mulDown(x.upper, y.upper),
                ),
                max4(
                    Arith.mulUp(x.lower, y.lower),
                    Arith.mulUp(x.lower, y.upper),
                    Arith.mulUp(x.upper, y.lower),
                    Arith.mulUp(x.upper, y.upper),
                ),
            );
        }

        /// Center of an interval. Typically used for bisection.
        pub inline fn median(x: Self) T {
            return Arith.median(x.lower, x.upper);
        }

        /// Intersection between two intervals (can be empty).
        pub inline fn intersect(x: Self, y: Self) Self {
            const l = Arith.max(x.lower, y.lower);
            const u = Arith.min(x.upper, y.upper);

            if (l <= u) {
                return init(l, u);
            } else {
                return empty();
            }
        }

        /// Division by an interval which does not contain 0. GPU-optimized implementation assuming
        /// division is expensive.
        inline fn divNonZero(x: Self, y: Self) Self {
            var xl: T = undefined;
            var yl: T = undefined;
            var xu: T = undefined;
            var yu: T = undefined;

            if (y.upper < 0) {
                xl = x.upper;
                xu = x.lower;
            } else {
                xl = x.lower;
                xu = x.upper;
            }

            if (x.upper < 0) {
                yl = y.lower;
                yu = y.upper;
            } else if (x.lower < 0) {
                if (y.upper < 0) {
                    yl = y.upper;
                    yu = y.upper;
                } else {
                    yl = y.lower;
                    yu = y.lower;
                }
            } else {
                yl = y.upper;
                yu = y.lower;
            }

            return init(Arith.divDown(xl, yl), Arith.divUp(xu, yu));
        }

        inline fn divPositive(x: Self, yu: T) Self {
            if (x.lower == 0 and x.upper == 0) return x;

            const xl = x.lower;
            const xu = x.upper;

            if (xu < 0) {
                return init(Arith.negInf(), Arith.divUp(xu, yu));
            } else if (xl < 0) {
                return init(Arith.negInf(), Arith.posInf());
            } else {
                return init(Arith.divDown(xl, yu), Arith.posInf());
            }
        }

        inline fn divNegative(x: Self, yl: T) Self {
            if (x.lower == 0 and x.upper == 0) return x;

            const xl = x.lower;
            const xu = x.upper;

            if (xu < 0) {
                return init(Arith.divDown(xu, yl), Arith.posInf());
            } else if (xl < 0) {
                return init(Arith.negInf(), Arith.posInf());
            } else {
                return init(Arith.negInf(), Arith.divUp(xl, yl));
            }
        }

        inline fn divZeroPart1(x: Self, y: Self) Division {
            if (x.lower == 0 and x.upper == 0) return .{ .part1 = x, .has_part2 = false };

            const xl = x.lower;
            const xu = x.upper;
            const yl = y.lower;
            const yu = y.upper;

            if (xu < 0) {
                return .{ .part1 = init(Arith.negInf(), Arith.divUp(xu, yu)), .has_part2 = true };
            } else if (xl < 0) {
                return .{ .part1 = init(Arith.negInf(), Arith.posInf()), .has_part2 = false };
            } else {
                return .{ .part1 = init(Arith.negInf(), Arith.divUp(xl, yl)), .has_part2 = true };
            }
        }

        inline fn divZeroPart2(x: Self, y: Self) Self {
            const xl = x.lower;
            const xu = x.upper;
            const yl = y.lower;
            const yu = y.upper;

            if (xu < 0) {
                return init(Arith.divDown(xu, yl), Arith.posInf());
            } else {
                return init(Arith.divDown(xl, yu), Arith.posInf());
            }
        }

        pub inline fn divisionPart1(x: Self, y: Self) Division {
            if (y.lower <= 0 and y.upper >= 0) {
                if (y.lower != 0) {
                    if (y.upper != 0) {
                        return divZeroPart1(x, y);
                    } else {
                        return .{ .part1 = divNegative(x, y.lower), .has_part2 = false };
                    }
                } else if (y.upper != 0) {
                    return .{ .part1 = divPositive(x, y.upper), .has_part2 = false };
                } else {
                    return .{ .part1 = empty(), .has_part2 = false };
                }
            } else {
                return .{ .part1 = divNonZero(x, y), .has_part2 = false };
            }
        }

        pub inline fn divisionPart2(x: Self, y: Self) Self {
            return divZeroPart2(x, y);
        }

        pub inline fn square(x: Self) Self {
            const xl = x.lower;
            const xu = x.upper;

            if (xl >= 0) {
                return init(Arith.mulDown(xl, xl), Arith.mulUp(xu, xu));
            } else if (xu <= 0) {
                return init(Arith.mulDown(xu, xu), Arith.mulUp(xl, xl));
            } else {
                return init(0, Arith.max(Arith.mulUp(xl, xl), Arith.mulUp(xu, xu)));
            }
        }
    };
}

inline fn min4(a: anytype, b: @TypeOf(a), c: @TypeOf(a), d: @TypeOf(a)) @TypeOf(a) {
    return @min(@min(a, b), @min(c, d));
}

inline fn max4(a: anytype, b: @TypeOf(a), c: @TypeOf(a), d: @TypeOf(a)) @TypeOf(a) {
    return @max(@max(a, b), @max(c, d));
}
