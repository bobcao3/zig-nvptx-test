const std = @import("std");
const builtin = @import("builtin");

pub const gpu = builtin.cpu.arch.isNvptx();
pub const device_cc: std.builtin.CallingConvention = if (gpu) .nvptx_device else .c;
pub const Out = if (gpu) [*]addrspace(.global) u32 else [*]u32;

/// A value that the compiler cannot see through on the GPU.
pub inline fn tid() u32 {
    return if (gpu) @workItemId(0) else 0;
}

/// Runs `run` as the kernel `kernel` on the GPU and prints the first `n` output words on the host.
pub fn Harness(comptime run: fn (Out) void, comptime n: usize) type {
    return struct {
        fn kernel(out: Out) callconv(if (gpu) .kernel else .auto) void {
            run(out);
        }

        comptime {
            if (gpu) @export(&kernel, .{ .name = "kernel" });
        }

        pub fn hostMain() void {
            var buf: [n]u32 = @splat(0xdeadbeef);
            run(&buf);
            for (buf) |word| std.debug.print("0x{x} ", .{word});
            std.debug.print("\n", .{});
        }
    };
}
