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

//! Type-specific implementation of rounded arithmetic operators. A thin layer over the directed
//! rounding instructions of PTX.

const std = @import("std");
const math = std.nvptx.math;

pub fn RoundedArith(comptime T: type) type {
    return struct {
        pub inline fn addDown(x: T, y: T) T {
            return math.add(T, .toward_negative_infinity, x, y);
        }

        pub inline fn addUp(x: T, y: T) T {
            return math.add(T, .toward_positive_infinity, x, y);
        }

        pub inline fn subDown(x: T, y: T) T {
            return addDown(x, -y);
        }

        pub inline fn subUp(x: T, y: T) T {
            return addUp(x, -y);
        }

        pub inline fn mulDown(x: T, y: T) T {
            return math.mul(T, .toward_negative_infinity, x, y);
        }

        pub inline fn mulUp(x: T, y: T) T {
            return math.mul(T, .toward_positive_infinity, x, y);
        }

        pub inline fn divDown(x: T, y: T) T {
            return math.div(T, .toward_negative_infinity, x, y);
        }

        pub inline fn divUp(x: T, y: T) T {
            return math.div(T, .toward_positive_infinity, x, y);
        }

        pub inline fn median(x: T, y: T) T {
            return (x + y) * 0.5;
        }

        pub inline fn sqrtDown(x: T) T {
            return math.sqrt(T, .toward_negative_infinity, x);
        }

        pub inline fn sqrtUp(x: T) T {
            return math.sqrt(T, .toward_positive_infinity, x);
        }

        pub inline fn intDown(x: T) T {
            return @floor(x);
        }

        pub inline fn intUp(x: T) T {
            return @ceil(x);
        }

        const inf = std.math.inf(T);
        const not_a_number = std.math.nan(T);

        pub inline fn negInf() T {
            return -inf;
        }

        pub inline fn posInf() T {
            return inf;
        }

        pub inline fn nan() T {
            return not_a_number;
        }

        pub inline fn min(x: T, y: T) T {
            return @min(x, y);
        }

        pub inline fn max(x: T, y: T) T {
            return @max(x, y);
        }
    };
}
