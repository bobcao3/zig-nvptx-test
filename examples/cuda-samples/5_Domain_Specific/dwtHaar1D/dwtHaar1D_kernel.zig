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

//! 1D DWT for Haar wavelet and signals with a length which is a power of 2.
//! The code reduces bank conflicts and non-coalesced reads / writes as
//! appropriate but does not fully remove them because the computational
//! overhead to achieve this would outweighs the benefit (see inline comments
//! for more details).
//! Large signals are subdivided into sub-signals with 512 elements and the
//! wavelet transform for these is computed with one block over 10 decomposition
//! levels. The resulting signal consisting of the approximation coefficients at
//! level X is then processed in a subsequent step on the device. This requires
//! interblock synchronization which is only possible on host side.
//! Detail coefficients which have been computed are not further referenced
//! during the decomposition so that they can be stored directly in their final
//! position in global memory. The transform and its storing scheme preserve
//! locality in the coefficients so that these writes are coalesced.
//! Approximation coefficients are stored in shared memory because they are
//! needed to compute the subsequent decomposition step. The top most
//! approximation coefficient for a sub-signal processed by one block is stored
//! in a special global memory location to simplify the processing after the
//! interblock synchronization.
//! Most books on wavelets explain the Haar wavelet decomposition. A good freely
//! available resource is the Wavelet primer by Stollnitz et al.
//! http://grail.cs.washington.edu/projects/wavelets/article/wavelet1.pdf
//! http://grail.cs.washington.edu/projects/wavelets/article/wavelet2.pdf
//! The basic of all Wavelet transforms is to decompose a signal into
//! approximation (a) and detail (d) coefficients where the detail tends to be
//! small or zero which allows / simplifies compression. The following "graphs"
//! demonstrate the transform for a signal
//! of length eight. The index always describes the decomposition level where
//! a coefficient arises. The input signal is interpreted as approximation signal
//! at level 0. The coefficients computed on the device are stored in the same
//! scheme as in the example. This data structure is particularly well suited for
//! compression and also preserves the hierarchical structure of the
//! decomposition.
//!
//!     -------------------------------------------------
//!     | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 |
//!     -------------------------------------------------
//!
//!     -------------------------------------------------
//!     | a_1 | a_1 | a_1 | a_1 | d_1 | d_1 | d_1 | d_1 |
//!     -------------------------------------------------
//!
//!     -------------------------------------------------
//!     | a_2 | a_2 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
//!     -------------------------------------------------
//!
//!     -------------------------------------------------
//!     | a_3 | d_3 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
//!     -------------------------------------------------
//!
//! Device Code.

const std = @import("std");
const common = @import("dwtHaar1D_common.zig");

extern var shared: [0]f32 align(16) addrspace(.shared);

export fn initValue(od: [*]addrspace(.global) f32, value: f32) callconv(.kernel) void {
    const index = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    od[index] = value;

    std.nvptx.barrierSyncAligned(.work_group, null);
}

/// Compute partial wavelet decomposition on the GPU using Haar basis. For each thread block the
/// full decomposition is computed but these results have to be combined. Use one thread to
/// perform the full decomposition.
///
/// `id` is the input data and `od` the output data. `approx_final` is the place to store the final
/// approximation coefficient for the subsignal. `dlevels` is the number of decomposition levels
/// for this transform. `slength_step_half` is half the signal length for the current
/// decomposition level (the offset for storing detail coefficients in global memory). `bdim` is
/// the block dimension.
export fn dwtHaar1D(
    id: [*]addrspace(.global) f32,
    od: [*]addrspace(.global) f32,
    approx_final: [*]addrspace(.global) f32,
    dlevels: u32,
    slength_step_half: u32,
    bdim: u32,
) callconv(.kernel) void {
    const shared_memory: [*]addrspace(.shared) f32 = &shared;

    const gdim = std.nvptx.nctaid(0);
    const bid = @workGroupId(0);
    const tid = @workItemId(0);

    const tid_global = (bid * bdim) + tid;
    const idata = (bid * (2 * bdim)) + tid;

    shared_memory[tid] = id[idata];
    shared_memory[tid + bdim] = id[idata + bdim];
    std.nvptx.barrierSyncAligned(.work_group, null);

    const data0 = shared_memory[2 * tid];
    const data1 = shared_memory[(2 * tid) + 1];
    std.nvptx.barrierSyncAligned(.work_group, null);

    od[tid_global + slength_step_half] = (data0 - data1) * common.inv_sqrt_2;

    const atid = tid + (tid >> common.log_num_banks);

    shared_memory[atid] = (data0 + data1) * common.inv_sqrt_2;

    std.nvptx.barrierSyncAligned(.work_group, null);

    if (dlevels > 1) {
        var offset_neighbor: u32 = 1;
        var num_threads = bdim >> 1;

        var idata0 = tid * 2;

        for (1..dlevels) |_| {
            if (tid < num_threads) {
                const idata1 = idata0 + offset_neighbor;

                const g_wpos = (num_threads * gdim) + (bid * num_threads) + tid;

                const c_idata0 = idata0 + (idata0 >> common.log_num_banks);
                const c_idata1 = idata1 + (idata1 >> common.log_num_banks);

                od[g_wpos] = (shared_memory[c_idata0] - shared_memory[c_idata1]) * common.inv_sqrt_2;

                shared_memory[c_idata0] = (shared_memory[c_idata0] + shared_memory[c_idata1]) * common.inv_sqrt_2;

                num_threads = num_threads >> 1;
                offset_neighbor <<= 1;
                idata0 = idata0 << 1;
            }

            std.nvptx.barrierSyncAligned(.work_group, null);
        }

        if (tid == 0) {
            approx_final[bid] = shared_memory[0];
        }
    }
}
