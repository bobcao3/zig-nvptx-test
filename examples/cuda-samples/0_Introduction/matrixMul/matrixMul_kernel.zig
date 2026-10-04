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

//! Matrix multiplication: C = A * B. Each work group computes one `block_size` x `block_size`
//! tile of C from tiles of A and B in shared memory.

const std = @import("std");

fn MatrixMulCuda(comptime block_size: i32) type {
    return struct {
        var as: [block_size][block_size]f32 addrspace(.shared) = undefined;
        var bs: [block_size][block_size]f32 addrspace(.shared) = undefined;

        /// `wa` is the width of A and `wb` is the width of B.
        fn matrixMulCuda(
            c: [*]addrspace(.global) f32,
            a: [*]addrspace(.global) const f32,
            b: [*]addrspace(.global) const f32,
            wa: i32,
            wb: i32,
        ) callconv(.kernel) void {
            const bx: i32 = @intCast(@workGroupId(0));
            const by: i32 = @intCast(@workGroupId(1));
            const tx: i32 = @intCast(@workItemId(0));
            const ty: i32 = @intCast(@workItemId(1));

            // The index of the first, the last and the step of the sub-matrices of A that the
            // work group processes.
            const a_begin = wa * block_size * by;
            const a_end = a_begin + wa - 1;
            const a_step = block_size;
            // The index of the first and the step of the sub-matrices of B.
            const b_begin = block_size * bx;
            const b_step = block_size * wb;

            var c_sub: f32 = 0;
            var a_index = a_begin;
            var b_index = b_begin;
            while (a_index <= a_end) : ({
                a_index += a_step;
                b_index += b_step;
            }) {
                as[@intCast(ty)][@intCast(tx)] = a[@intCast(a_index + wa * ty + tx)];
                bs[@intCast(ty)][@intCast(tx)] = b[@intCast(b_index + wb * ty + tx)];
                std.nvptx.barrierSyncAligned(.work_group, null);

                inline for (0..block_size) |k| {
                    c_sub += as[@intCast(ty)][k] * bs[k][@intCast(tx)];
                }
                std.nvptx.barrierSyncAligned(.work_group, null);
            }

            const c_index = wb * block_size * by + block_size * bx;
            c[@intCast(c_index + wb * ty + tx)] = c_sub;
        }

        comptime {
            @export(&matrixMulCuda, .{ .name = std.fmt.comptimePrint("matrixMulCuda{d}", .{block_size}) });
        }
    };
}

comptime {
    _ = MatrixMulCuda(16);
    _ = MatrixMulCuda(32);
}
