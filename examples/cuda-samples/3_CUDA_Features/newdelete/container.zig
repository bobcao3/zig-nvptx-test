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

/// The parent type of the containers. A container has a table of functions in place of virtual
/// functions.
pub fn Container(comptime T: type) type {
    return struct {
        vtable: *const VTable,

        const Self = @This();

        pub const VTable = struct {
            push: *const fn (container: *Self, element: T) void,
            pop: *const fn (container: *Self, element: *T) bool,
            destroy: *const fn (container: *Self) void,
        };

        pub inline fn push(container: *Self, element: T) void {
            container.vtable.push(container, element);
        }

        pub inline fn pop(container: *Self, element: *T) bool {
            return container.vtable.pop(container, element);
        }

        /// Runs the destructor of the container and frees the memory of the container itself,
        /// like `delete` does.
        pub inline fn destroy(container: *Self) void {
            container.vtable.destroy(container);
        }
    };
}

/// A container that uses linear memory as data storage.
///
/// NOTE: This education purpose implementation has restricted functionality. For example,
/// concurrent push and pop operations will not work correctly.
pub fn Vector(comptime T: type) type {
    return struct {
        container: Container(T),
        data: ?[*]T,
        top: i32,

        const Self = @This();

        const vtable: Container(T).VTable = .{
            .push = push,
            .pop = pop,
            .destroy = destroy,
        };

        /// Allocates the data on the heap. Only one work item may call this function.
        pub inline fn init(max_size: i32) Self {
            const bytes = @as(usize, @intCast(max_size)) * @sizeOf(T);
            const data: ?[*]T = @ptrCast(@alignCast(std.nvptx.malloc(bytes)));
            return .{ .container = .{ .vtable = &vtable }, .data = data, .top = -1 };
        }

        /// Uses a preallocated buffer as data.
        pub inline fn initPreallocated(preallocated_buffer: [*]T) Self {
            return .{ .container = .{ .vtable = &vtable }, .data = preallocated_buffer, .top = -1 };
        }

        /// Allocates a vector on the heap, like `new`.
        pub inline fn create(max_size: i32) *Self {
            const vector: *Self = @ptrCast(@alignCast(std.nvptx.malloc(@sizeOf(Self))));
            vector.* = .init(max_size);
            return vector;
        }

        fn push(container: *Container(T), element: T) void {
            const vector: *Self = @fieldParentPtr("container", container);
            if (vector.data) |data| {
                const index = @atomicRmw(i32, &vector.top, .Add, 1, .monotonic);
                data[@intCast(index + 1)] = element;
            }
        }

        fn pop(container: *Container(T), element: *T) bool {
            const vector: *Self = @fieldParentPtr("container", container);
            if (vector.data) |data| {
                if (@atomicLoad(i32, &vector.top, .monotonic) >= 0) {
                    const index = @atomicRmw(i32, &vector.top, .Sub, 1, .monotonic);
                    if (index >= 0) {
                        element.* = data[@intCast(index)];
                        return true;
                    }
                }
            }
            return false;
        }

        /// Only one work item may call this function.
        fn destroy(container: *Container(T)) void {
            const vector: *Self = @fieldParentPtr("container", container);
            if (vector.data) |data| std.nvptx.free(data);
            std.nvptx.free(vector);
        }
    };
}
