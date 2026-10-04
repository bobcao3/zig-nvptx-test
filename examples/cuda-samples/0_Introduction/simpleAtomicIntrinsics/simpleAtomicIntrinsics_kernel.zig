// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

//! A simple program demonstrating why atomic operations are needed, using the atomic
//! intrinsics atomicAdd, atomicMax, and atomicCAS.
//!
//! 1,000,000 work items write into an array of only 10 integers (each work item maps to a slot
//! with index % ARRAY_SIZE), so ~100,000 work items update every element at the same time. A
//! plain update like g[i] = g[i] + 1 is three separate steps: read the value, modify it, write it
//! back. When two work items interleave (both read the same old value, both write back the same
//! result), one update silently overwrites the other and is lost. This is a race condition: the
//! result depends on timing, not program logic, and changes every run.
//!
//! Each operation is therefore run four ways: a non-atomic kernel that loses most of its updates
//! to these race conditions, and three atomic kernels that do the read-modify-write as one
//! indivisible operation and give the exact result: first with the atomic builtins, then with
//! the sequentially consistent order of `cuda::std::atomic_ref`, and finally with the order of
//! `cuda::atomic_ref`.
const num_threads = 1000000;
const array_size = 10;

inline fn globalIndex() i32 {
    return @intCast(@workGroupId(0) * @workGroupSize(0) + @workItemId(0));
}

/// Increment without atomics: work items race on shared elements and lose updates.
export fn increment(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    // The grid is rounded up, so the guard ignores the extra work items.
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));
        g[i] = g[i] + 1;
    }
}

/// Increment with an atomic add: the read-modify-write is one indivisible operation.
export fn incrementAtomic(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));
        _ = @atomicRmw(i32, &g[i], .Add, 1, .monotonic);
    }
}

/// Increment with a sequentially consistent atomic add, the default order of `std::atomic_ref`.
export fn incrementAtomicStd(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));
        _ = @atomicRmw(i32, &g[i], .Add, 1, .seq_cst);
    }
}

/// Increment with a sequentially consistent atomic add, the default order of
/// `cuda::atomic_ref`.
export fn incrementAtomicCuda(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));
        _ = @atomicRmw(i32, &g[i], .Add, 1, .seq_cst);
    }
}

/// Max without atomics: another work item can write between the compare and the store, so a
/// smaller value can overwrite a larger one.
export fn max(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    // Each work item contributes its global index as the value.
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));
        if (g[i] < tid) g[i] = tid;
    }
}

/// Max with an atomic max: the compare and the store happen as one operation.
export fn maxAtomic(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));
        _ = @atomicRmw(i32, &g[i], .Max, tid, .monotonic);
    }
}

/// Max with a compare-exchange loop that stops as soon as the stored value is already >= tid.
export fn maxAtomicStd(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));

        var expected = @atomicLoad(i32, &g[i], .seq_cst);
        while (expected < tid) {
            expected = @cmpxchgWeak(i32, &g[i], expected, tid, .seq_cst, .seq_cst) orelse break;
        }
    }
}

/// Max with the same compare-exchange loop as `maxAtomicStd`.
export fn maxAtomicCuda(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));

        var expected = @atomicLoad(i32, &g[i], .seq_cst);
        while (expected < tid) {
            expected = @cmpxchgWeak(i32, &g[i], expected, tid, .seq_cst, .seq_cst) orelse break;
        }
    }
}

/// Increment written as compare-and-swap, without atomics: the value can change between the
/// compare and the swap, so increments are lost.
export fn cas(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));

        const expected = g[i]; // read
        if (g[i] == expected) // compare
            g[i] = expected + 1; // swap (not atomic with the compare)
    }
}

/// Increment with a compare-and-swap retry loop, the classic pattern for building any atomic
/// operation. A failed swap returns the value that it found, and the loop retries with it.
export fn casAtomic(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));

        var old = g[i];
        var assumed: i32 = undefined;
        while (true) {
            assumed = old;
            old = @cmpxchgStrong(i32, &g[i], assumed, assumed + 1, .monotonic, .monotonic) orelse assumed;
            if (old == assumed) break;
        }
    }
}

/// Increment with a weak compare-exchange loop. On failure, `expected` holds the value that the
/// exchange found, so the loop retries with fresh data.
export fn casAtomicStd(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));

        var expected = @atomicLoad(i32, &g[i], .seq_cst);
        while (@cmpxchgWeak(i32, &g[i], expected, expected + 1, .seq_cst, .seq_cst)) |found| {
            expected = found;
        }
    }
}

/// Increment with the same weak compare-exchange loop as `casAtomicStd`.
export fn casAtomicCuda(g: [*]addrspace(.global) i32) callconv(.kernel) void {
    const tid = globalIndex();
    if (tid < num_threads) {
        const i: usize = @intCast(@mod(tid, array_size));

        var expected = @atomicLoad(i32, &g[i], .seq_cst);
        while (@cmpxchgWeak(i32, &g[i], expected, expected + 1, .seq_cst, .seq_cst)) |found| {
            expected = found;
        }
    }
}
