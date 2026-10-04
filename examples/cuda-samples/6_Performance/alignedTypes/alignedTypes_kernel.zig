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

//! Copy kernels that show the access speed gap between aligned and misaligned structures. Each
//! kernel copies the elements of one type.

const std = @import("std");
const common = @import("alignedTypes_common.zig");

fn TestKernel(comptime kind: common.Kind) type {
    const Element = kind.Element();
    return struct {
        /// Copies on a per-element basis, so padded structures are not copied byte by byte.
        fn testKernel(
            d_odata: [*]addrspace(.global) Element,
            d_idata: [*]addrspace(.global) const Element,
            num_elements: i32,
        ) callconv(.kernel) void {
            const tid = @workGroupSize(0) * @workGroupId(0) + @workItemId(0);
            const num_threads = @workGroupSize(0) * std.nvptx.nctaid(0);

            var pos = tid;
            while (pos < num_elements) : (pos += num_threads) {
                d_odata[pos] = d_idata[pos];
            }
        }

        comptime {
            @export(&testKernel, .{ .name = "testKernel_" ++ @tagName(kind) });
        }
    };
}

comptime {
    for (std.enums.values(common.Kind)) |kind| _ = TestKernel(kind);
}
