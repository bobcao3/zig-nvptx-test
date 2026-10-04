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

//! This sample demonstrates dynamic global memory allocation through device `malloc` and `free`
//! and virtual function declarations available with CUDA 4.0.

const std = @import("std");
const container = @import("container.zig");
const Container = container.Container;
const Vector = container.Vector;

inline fn blockSync() void {
    std.nvptx.barrierSyncAligned(.work_group, null);
}

/// The Vector object and the data storage are allocated in device heap memory. This makes it
/// persistent for the lifetime of the CUDA context. The grid has only one thread as only a single
/// object instance is needed.
export fn vectorCreate(g_container: *addrspace(.global) *Container(i32), max_size: i32) callconv(.kernel) void {
    g_container.* = &Vector(i32).create(max_size).container;
}

/// All threads of the grid cooperatively populate the shared Container object with data.
export fn containerFill(g_container: *addrspace(.global) *Container(i32)) callconv(.kernel) void {
    if (@workItemId(0) == 0) {
        g_container.*.push(@intCast(@workGroupId(0)));
    }
}

/// All threads of the grid cooperatively consume the data from the shared Container object.
export fn containerConsume(
    g_container: *addrspace(.global) *Container(i32),
    d_result: [*]addrspace(.global) i32,
) callconv(.kernel) void {
    const idx = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    var v: i32 = undefined;

    if (g_container.*.pop(&v)) {
        d_result[idx] = v;
    } else {
        d_result[idx] = -1;
    }
}

export fn containerDelete(g_container: *addrspace(.global) *Container(i32)) callconv(.kernel) void {
    g_container.*.destroy();
}

const ComplexType = extern struct {
    a: i32,
    b: i32,
    c: f32,
    d: f32,
};

fn PlacementNew(comptime T: type) type {
    return struct {
        var s_buffer: [@sizeOf(Vector(T))]u8 align(8) addrspace(.shared) = undefined;
        var s_data: [1024]T align(8) addrspace(.shared) = undefined;
        var s_vector: *Vector(T) addrspace(.shared) = undefined;
    };
}

/// Uses placement new to put the shared Vector object and the data in shared memory.
export fn placementNew(d_result: [*]addrspace(.global) i32) callconv(.kernel) void {
    const shared = PlacementNew(i32);

    if (@workItemId(0) == 0) {
        const vector: *Vector(i32) = @ptrCast(@addrSpaceCast(&shared.s_buffer));
        vector.* = .initPreallocated(@as(*[1024]i32, @addrSpaceCast(&shared.s_data)));
        shared.s_vector = vector;
    }

    blockSync();

    if (@workItemId(0) & 1 == 0) {
        shared.s_vector.container.push(@intCast(@workItemId(0) >> 1));
    }

    blockSync();

    var v: i32 = undefined;

    if (shared.s_vector.container.pop(&v)) {
        d_result[@workItemId(0)] = v;
    } else {
        d_result[@workItemId(0)] = -1;
    }
}

export fn complexVector(d_result: [*]addrspace(.global) i32) callconv(.kernel) void {
    const shared = PlacementNew(ComplexType);

    if (@workItemId(0) == 0) {
        const vector: *Vector(ComplexType) = @ptrCast(@addrSpaceCast(&shared.s_buffer));
        vector.* = .initPreallocated(@as(*[1024]ComplexType, @addrSpaceCast(&shared.s_data)));
        shared.s_vector = vector;
    }

    blockSync();

    if (@workItemId(0) & 1 == 0) {
        const data: ComplexType = .{
            .a = @intCast(@workItemId(0) >> 1),
            .b = @intCast(@workGroupId(0)),
            .c = @as(f32, @floatFromInt(@workItemId(0))) / @as(f32, @floatFromInt(@workGroupSize(0))),
            .d = @as(f32, @floatFromInt(@workGroupId(0))) / @as(f32, @floatFromInt(std.nvptx.nctaid(0))),
        };

        shared.s_vector.container.push(data);
    }

    blockSync();

    var v: ComplexType = undefined;

    if (shared.s_vector.container.pop(&v)) {
        d_result[@workItemId(0)] = v.a;
    } else {
        d_result[@workItemId(0)] = -1;
    }
}
