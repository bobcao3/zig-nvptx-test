const builtin = @import("builtin");
const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;
const Status = std.zig.Server.Message.TestResults.Status;
const Cuda = @import("cuda");
const Sm = @import("Sm");

const usage =
    \\Usage: zig-nvptx-test-executor [--ptxas=<path>] <test.ptx> [--listen=-]
    \\
    \\Runs the tests of a `zig test -target nvptx64-cuda` module on the first CUDA
    \\device. The module must use src/test_runner.zig of this repository as its test runner.
    \\Pass the executor to `zig test` with `--test-cmd <executor> --test-cmd-bin`.
    \\
    \\With --ptxas, or with the environment variable PTXAS, the executor first checks the module
    \\with that ptxas for the target that the module names. When the device cannot load that target,
    \\the executor prints `skipped (needs <target>)` instead of running the tests, and without a CUDA
    \\driver it prints `skipped (no CUDA driver)`. Both exit with 0.
    \\
;

const Device = struct {
    cuda: Cuda.Raw,
    kernels: [kernel_names.len]Cuda.Handle,
    /// Holds the status word, then the text length, then the text.
    memory: Cuda.DevicePointer,

    const kernel_names = .{ "zig_test_count", "zig_test_name", "zig_test_run" };
    const text_capacity = 4096;
    const memory_size = 8 + text_capacity;
    const stack_size = 16 * 1024;

    fn init(cuda: Cuda.Raw, ptx: [:0]const u8, ptx_path: [:0]const u8) Device {
        cuda.check(cuda.cuCtxSetLimit(.stack_size, stack_size), "cuCtxSetLimit");
        const module = cuda.loadModule(ptx, ptx_path, &.{});

        var kernels: [kernel_names.len]Cuda.Handle = undefined;
        inline for (kernel_names, &kernels) |name, *kernel| {
            cuda.check(cuda.cuModuleGetFunction(kernel, module, name), "cuModuleGetFunction");
        }
        var memory: Cuda.DevicePointer = undefined;
        cuda.check(cuda.cuMemAlloc_v2(&memory, memory_size), "cuMemAlloc");
        return .{ .cuda = cuda, .kernels = kernels, .memory = memory };
    }

    /// Returns the CUDA error name if the kernel did not finish, for example after a trap.
    fn launch(device: *const Device, kernel: usize, params: []const *const anyopaque) ?[]const u8 {
        const cuda = &device.cuda;
        var result = cuda.cuLaunchKernel(device.kernels[kernel], 1, 1, 1, 1, 1, 1, 0, null, params.ptr, null);
        if (result == 0) result = cuda.cuCtxSynchronize();
        if (result == 0) return null;
        return cuda.errorName(result);
    }

    fn testCount(device: *const Device) u32 {
        if (device.launch(0, &.{&device.memory})) |err| fatal("zig_test_count failed: {s}", .{err});
        var count: u32 = undefined;
        device.copyOut(std.mem.asBytes(&count), 0);
        return count;
    }

    fn testName(device: *const Device, index: u32, buffer: *[text_capacity]u8) []const u8 {
        const len_pointer = device.memory + 4;
        const text_pointer = device.memory + 8;
        const capacity: u32 = text_capacity;
        if (device.launch(1, &.{ &index, &len_pointer, &text_pointer, &capacity })) |err| {
            fatal("zig_test_name failed: {s}", .{err});
        }
        return device.text(buffer);
    }

    const Outcome = union(enum) {
        pass,
        skip,
        fail: []const u8,
        trap: []const u8,
    };

    fn runTest(device: *const Device, index: u32, buffer: *[text_capacity]u8) Outcome {
        const len_pointer = device.memory + 4;
        const text_pointer = device.memory + 8;
        const capacity: u32 = text_capacity;
        if (device.launch(2, &.{ &index, &device.memory, &len_pointer, &text_pointer, &capacity })) |err| {
            return .{ .trap = err };
        }
        var status: u32 = undefined;
        device.copyOut(std.mem.asBytes(&status), 0);
        return switch (std.enums.fromInt(Status, status) orelse fatal("invalid test status {d}", .{status})) {
            .pass => .pass,
            .skip => .skip,
            .fail => .{ .fail = device.text(buffer) },
        };
    }

    fn text(device: *const Device, buffer: *[text_capacity]u8) []const u8 {
        var len: u32 = undefined;
        device.copyOut(std.mem.asBytes(&len), 4);
        if (len > text_capacity) fatal("text of {d} bytes does not fit in {d} bytes", .{ len, text_capacity });
        device.copyOut(buffer[0..len], 8);
        return buffer[0..len];
    }

    fn copyOut(device: *const Device, destination: []u8, offset: u64) void {
        if (destination.len == 0) return;
        device.cuda.check(
            device.cuda.cuMemcpyDtoH_v2(destination.ptr, device.memory + offset, destination.len),
            "cuMemcpyDtoH",
        );
    }
};

const Counts = struct {
    next: u32 = 0,
    ok: u32 = 0,
    skip: u32 = 0,
    fail: u32 = 0,

    fn parse(text: []const u8) ?Counts {
        var counts: Counts = .{};
        var it = std.mem.splitScalar(u8, text, ',');
        inline for (.{ "next", "ok", "skip", "fail" }) |name| {
            @field(counts, name) = std.fmt.parseUnsigned(u32, it.next() orelse return null, 10) catch return null;
        }
        return if (it.next() == null) counts else null;
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    var args = try init.minimal.args.iterateAllocator(arena);
    _ = args.skip();
    var ptx_path: ?[:0]const u8 = null;
    var ptxas: ?[]const u8 = init.environ_map.get("PTXAS");
    var listen = false;
    var counts: Counts = .{};
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--listen=-")) {
            listen = true;
        } else if (std.mem.cutPrefix(u8, arg, "--ptxas=")) |path| {
            ptxas = path;
        } else if (std.mem.cutPrefix(u8, arg, "--resume=")) |text| {
            counts = Counts.parse(text) orelse fatal("invalid argument: {s}", .{arg});
        } else if (std.mem.startsWith(u8, arg, "--seed=") or std.mem.startsWith(u8, arg, "--cache-dir=")) {
            // The build runner passes these to every test runner. The executor does not use them.
        } else if (ptx_path == null and !std.mem.startsWith(u8, arg, "--")) {
            ptx_path = arg;
        } else {
            fatal("unrecognized argument: {s}\n{s}", .{ arg, usage });
        }
    }
    const path = ptx_path orelse fatal(usage, .{});

    const ptx = try Io.Dir.cwd().readFileAllocOptions(io, path, arena, .unlimited, .of(u8), 0);
    const sm = Sm.fromPtx(ptx) orelse fatal("{s} has no .target directive", .{path});
    if (counts.next == 0) if (ptxas) |ptxas_path| try validate(io, arena, ptxas_path, path, sm);
    const cuda = Cuda.Raw.load() catch {
        std.debug.print("skipped (no CUDA driver)\n", .{});
        return;
    };
    cuda.initDevice();
    const capability = cuda.computeCapability();
    if (!sm.loadsOn(capability.major, capability.minor)) {
        std.debug.print("skipped (needs {f})\n", .{sm});
        return;
    }
    const device: Device = .init(cuda, ptx, path);

    if (listen) {
        return mainServer(io, arena, &device);
    } else {
        return mainTerminal(io, arena, &device, path, counts);
    }
}

/// Runs ptxas on the module for the architecture of its `.target` directive, so that the report
/// names the line that ptxas rejects.
fn validate(io: Io, arena: std.mem.Allocator, ptxas: []const u8, path: []const u8, sm: Sm) !void {
    const cubin_path = try std.fmt.allocPrint(arena, "{s}.cubin", .{path});
    defer Io.Dir.cwd().deleteFile(io, cubin_path) catch {};
    const arch_arg = try std.fmt.allocPrint(arena, "-arch={f}", .{sm});
    const result = std.process.run(arena, io, .{
        .argv = &.{ ptxas, arch_arg, path, "-o", cubin_path },
    }) catch |err| fatal("unable to run {s}: {t}", .{ ptxas, err });
    if (!result.term.success()) fatal("ptxas failed:\n{s}{s}", .{ result.stdout, result.stderr });
}

fn mainServer(io: Io, arena: std.mem.Allocator, device: *const Device) !void {
    var stdin_buffer: [4096]u8 = undefined;
    var stdout_buffer: [4096]u8 = undefined;
    var stdin_reader: Io.File.Reader = .initStreaming(.stdin(), io, &stdin_buffer);
    var stdout_writer: Io.File.Writer = .initStreaming(.stdout(), io, &stdout_buffer);
    var server: std.zig.Server = .{
        .in = &stdin_reader.interface,
        .out = &stdout_writer.interface,
    };
    try server.serveStringMessage(.zig_version, builtin.zig_version_string);

    var buffer: [Device.text_capacity]u8 = undefined;
    while (true) {
        const hdr = try server.receiveMessage();
        switch (hdr.tag) {
            .exit => return std.process.exit(0),
            .query_test_metadata => {
                const count = device.testCount();
                var string_bytes: std.ArrayList(u8) = .empty;
                try string_bytes.append(arena, 0); // Reserve 0 for null.
                const names = try arena.alloc(u32, count);
                const expected_panic_msgs = try arena.alloc(u32, count);
                for (names, expected_panic_msgs, 0..) |*name, *expected_panic_msg, index| {
                    name.* = @intCast(string_bytes.items.len);
                    try string_bytes.appendSlice(arena, device.testName(@intCast(index), &buffer));
                    try string_bytes.append(arena, 0);
                    expected_panic_msg.* = 0;
                }
                try server.serveTestMetadata(.{
                    .names = names,
                    .expected_panic_msgs = expected_panic_msgs,
                    .string_bytes = string_bytes.items,
                });
            },
            .run_test => {
                const index = try server.receiveBody_u32();
                try server.serveStringMessage(.test_started, &.{});
                const status: Status = switch (device.runTest(index, &buffer)) {
                    .pass => .pass,
                    .skip => .skip,
                    .fail => |error_name| status: {
                        std.debug.print("error.{s}\n", .{error_name});
                        break :status .fail;
                    },
                    // Exiting lets the build runner report the crash against this test and continue
                    // with the remaining tests.
                    .trap => |err| fatal("trap: {s}", .{err}),
                };
                try server.serveTestResults(.{
                    .index = index,
                    .flags = .{
                        .status = status,
                        .fuzz = false,
                        .log_err_count = 0,
                        .leak_count = 0,
                    },
                });
            },
            else => {
                std.debug.print("unsupported message: {x}\n", .{@backingInt(hdr.tag)});
                std.process.exit(1);
            },
        }
    }
}

fn mainTerminal(
    io: Io,
    arena: std.mem.Allocator,
    device: *const Device,
    ptx_path: []const u8,
    initial_counts: Counts,
) !void {
    const count = device.testCount();
    var counts = initial_counts;
    var buffer: [Device.text_capacity]u8 = undefined;
    while (counts.next < count) {
        const index = counts.next;
        counts.next += 1;
        std.debug.print("{d}/{d} {s}...", .{ index + 1, count, device.testName(index, &buffer) });
        switch (device.runTest(index, &buffer)) {
            .pass => {
                counts.ok += 1;
                std.debug.print("OK\n", .{});
            },
            .skip => {
                counts.skip += 1;
                std.debug.print("SKIP\n", .{});
            },
            .fail => |error_name| {
                counts.fail += 1;
                std.debug.print("FAIL ({s})\n", .{error_name});
            },
            .trap => |err| {
                counts.fail += 1;
                std.debug.print("FAIL (trap: {s})\n", .{err});
                if (counts.next == count) break;
                // A trap leaves the CUDA context unusable, so a new process runs the remaining tests.
                const resume_arg = try std.fmt.allocPrint(arena, "--resume={d},{d},{d},{d}", .{
                    counts.next, counts.ok, counts.skip, counts.fail,
                });
                const self_exe = try std.process.executablePathAlloc(io, arena);
                var child = try std.process.spawn(io, .{ .argv = &.{ self_exe, ptx_path, resume_arg } });
                const term = try child.wait(io);
                std.process.exit(switch (term) {
                    .exited => |code| code,
                    else => 1,
                });
            },
        }
    }
    if (counts.ok == count) {
        std.debug.print("All {d} tests passed.\n", .{counts.ok});
    } else {
        std.debug.print("{d} passed; {d} skipped; {d} failed.\n", .{ counts.ok, counts.skip, counts.fail });
    }
    if (counts.fail != 0) std.process.exit(1);
}
