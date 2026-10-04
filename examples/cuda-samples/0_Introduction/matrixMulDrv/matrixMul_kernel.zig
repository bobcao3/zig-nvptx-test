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

//! Matrix multiplication: C = A * B. Device code.

const std = @import("std");

fn MatrixMul(comptime block_size: usize) type {
    return struct {
        var as: [block_size][block_size]f32 addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 addrspace(.shared) = undefined;

        /// `wa` is the width of A and `wb` is the width of B.
        fn matrixMul(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) f32,
            b: [*]addrspace(.global) f32,
            wa: usize,
            wb: usize,
        ) callconv(.kernel) void {
            const bx: usize = @workGroupId(0);
            const by: usize = @workGroupId(1);
            const tx: usize = @workItemId(0);
            const ty: usize = @workItemId(1);

            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0;
            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                as[ty][tx] = a[a_index + wa * ty + tx];
                bs[ty][tx] = b[b_index + wb * ty + tx];
                std.nvptx.barrierSyncAligned(.work_group, null);

                inline for (0..block_size) |k| {
                    c_sub += as[ty][k] * bs[k][tx];
                }
                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[c_index + wb * ty + tx] = c_sub;
        }

        comptime {
            @export(&matrixMul, .{
                .name = std.fmt.comptimePrint("matrixMul_bs{d}_64bit", .{block_size}),
            });
        }
    };
}

comptime {
    _ = MatrixMul(8);
    _ = MatrixMul(16);
    _ = MatrixMul(32);
}
