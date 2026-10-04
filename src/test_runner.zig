//! Test runner for `zig test -target nvptx64-cuda --test-runner <this file>`.
//! `std.start` exports no entry point for CUDA, so the executor runs the tests on the device by
//! launching the kernels below.

const builtin = @import("builtin");
const std = @import("std");
const Status = std.zig.Server.Message.TestResults.Status;

comptime {
    @export(&testCount, .{ .name = "zig_test_count" });
    @export(&testName, .{ .name = "zig_test_name" });
    @export(&testRun, .{ .name = "zig_test_run" });
}

fn testCount(count: *addrspace(.global) u32) callconv(.kernel) void {
    count.* = @intCast(builtin.test_functions.len);
}

fn testName(
    index: u32,
    len: *addrspace(.global) u32,
    buffer: [*]addrspace(.global) u8,
    capacity: u32,
) callconv(.kernel) void {
    copyOut(builtin.test_functions[index].name, len, buffer, capacity);
}

fn testRun(
    index: u32,
    status: *addrspace(.global) u32,
    len: *addrspace(.global) u32,
    buffer: [*]addrspace(.global) u8,
    capacity: u32,
) callconv(.kernel) void {
    if (builtin.test_functions[index].func()) |_| {
        status.* = @backingInt(Status.pass);
        len.* = 0;
    } else |err| switch (err) {
        error.SkipZigTest => {
            status.* = @backingInt(Status.skip);
            len.* = 0;
        },
        else => {
            status.* = @backingInt(Status.fail);
            copyOut(@errorName(err), len, buffer, capacity);
        },
    }
}

/// Stores the full length of `text` in `len` even when `capacity` truncates the copy, so that the
/// host can detect the truncation.
fn copyOut(
    text: []const u8,
    len: *addrspace(.global) u32,
    buffer: [*]addrspace(.global) u8,
    capacity: u32,
) void {
    len.* = @intCast(text.len);
    for (text[0..@min(text.len, capacity)], buffer) |byte, *out| out.* = byte;
}
