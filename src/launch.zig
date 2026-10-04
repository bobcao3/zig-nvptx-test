const builtin = @import("builtin");
const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;
const Status = std.zig.Server.Message.TestResults.Status;
const Cuda = @import("cuda");
const Sm = @import("Sm");

const usage =
    \\Usage: zig-nvptx-launch <module.ptx> [--link=<other.ptx>]... <case>... [--listen=-]
    \\
    \\Loads a PTX module on the first CUDA device, runs each case and checks its results.
    \\--link links more PTX files into the module, for example the definitions of its extern
    \\functions. Without a CUDA driver, or when the device cannot load the target of the module,
    \\every case is skipped.
    \\
    \\Cases:
    \\  saxpy                                    checks the kernel `saxpy`
    \\  divmod                                   checks the kernel `divmod`
    \\  tma                                      checks the kernel `tmaTile` with a tensor map
    \\  expect <kernel> <words>                  runs kernel(out) with one thread
    \\  contend <kernel> <grid> <block> <words>  runs kernel(out) on grid x block threads
    \\  trap <kernel>                            expects kernel(out) to fail with a launch failure
    \\
    \\<words> is one argument that lists the expected values of out[0], out[1], ... separated by
    \\spaces, or `@<path>` to read the list from a file. A word is decimal, 0x hexadecimal, a
    \\float with a '.', or '-' to skip that word. For `contend`, a word i<index>=<value> stores
    \\<value> in out[index] before the launch; the other words start at 0. For `expect`, all
    \\words start at 0xdeadbeef.
    \\
;

const Case = struct {
    kind: Kind,
    kernel: [:0]const u8,
    grid: u32 = 1,
    block: u32 = 1,
    words: []const u8 = "",

    const Kind = enum { saxpy, divmod, tma, expect, contend, trap };
};

const Expected = struct {
    initial: [word_count]u32,
    want: [word_count]u32,
    check: [word_count]bool,
    len: usize,

    fn parse(text: []const u8, initial_word: u32) Expected {
        var expected: Expected = .{
            .initial = @splat(initial_word),
            .want = undefined,
            .check = @splat(false),
            .len = 0,
        };
        var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
        while (it.next()) |word| {
            if (word[0] == 'i') if (std.mem.findScalar(u8, word, '=')) |equals| {
                const index = parseWord(word[1..equals]);
                if (index >= word_count) fatal("invalid initial value: {s}", .{word});
                expected.initial[index] = parseWord(word[equals + 1 ..]);
                continue;
            };
            if (expected.len == word_count) fatal("more than {d} words", .{word_count});
            if (!std.mem.eql(u8, word, "-")) {
                expected.want[expected.len] = parseWord(word);
                expected.check[expected.len] = true;
            }
            expected.len += 1;
        }
        return expected;
    }

    fn parseWord(word: []const u8) u32 {
        if (std.mem.findScalar(u8, word, '.') != null) {
            const float = std.fmt.parseFloat(f32, word) catch fatal("invalid word: {s}", .{word});
            return @bitCast(float);
        }
        return std.fmt.parseInt(u32, word, 0) catch fatal("invalid word: {s}", .{word});
    }
};

const word_count = 1024;

const Device = struct {
    cuda: Cuda.Raw,
    module: Cuda.Handle,

    fn function(device: *const Device, name: [:0]const u8) Cuda.Handle {
        var handle: Cuda.Handle = null;
        device.cuda.check(device.cuda.cuModuleGetFunction(&handle, device.module, name), "cuModuleGetFunction");
        return handle;
    }

    fn alloc(device: *const Device, size: usize) Cuda.DevicePointer {
        var pointer: Cuda.DevicePointer = undefined;
        device.cuda.check(device.cuda.cuMemAlloc_v2(&pointer, size), "cuMemAlloc");
        return pointer;
    }

    fn copyIn(device: *const Device, destination: Cuda.DevicePointer, source: []const u8) void {
        device.cuda.check(device.cuda.cuMemcpyHtoD_v2(destination, source.ptr, source.len), "cuMemcpyHtoD");
    }

    fn copyOut(device: *const Device, destination: []u8, source: Cuda.DevicePointer) void {
        device.cuda.check(device.cuda.cuMemcpyDtoH_v2(destination.ptr, source, destination.len), "cuMemcpyDtoH");
    }

    fn launch(
        device: *const Device,
        name: [:0]const u8,
        grid: u32,
        block: u32,
        params: []const *const anyopaque,
    ) Cuda.Result {
        const cuda = &device.cuda;
        const result = cuda.cuLaunchKernel(device.function(name), grid, 1, 1, block, 1, 1, 0, null, params.ptr, null);
        if (result != 0) return result;
        return cuda.cuCtxSynchronize();
    }

    fn launchOrFail(device: *const Device, name: [:0]const u8, grid: u32, block: u32, params: []const *const anyopaque) void {
        device.cuda.check(device.launch(name, grid, block, params), "cuLaunchKernel");
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);
    var ptx_path: ?[:0]const u8 = null;
    var link_paths: std.ArrayList([:0]const u8) = .empty;
    var cases: std.ArrayList(Case) = .empty;
    var listen = false;
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--listen=-")) {
            listen = true;
        } else if (std.mem.startsWith(u8, arg, "--seed=") or std.mem.startsWith(u8, arg, "--cache-dir=")) {
            // The build runner passes these to every test runner. The launcher does not use them.
        } else if (std.mem.startsWith(u8, arg, "--link=")) {
            try link_paths.append(arena, arg["--link=".len..]);
        } else if (ptx_path == null) {
            ptx_path = arg;
        } else {
            const kind = std.meta.stringToEnum(Case.Kind, arg) orelse fatal("unknown case {s}\n{s}", .{ arg, usage });
            const operands: usize = switch (kind) {
                .saxpy, .divmod, .tma => 0,
                .trap => 1,
                .expect => 2,
                .contend => 4,
            };
            if (index + operands >= args.len) fatal("missing operands of {s}\n{s}", .{ arg, usage });
            const operand = args[index + 1 ..][0..operands];
            index += operands;
            try cases.append(arena, switch (kind) {
                .saxpy => .{ .kind = kind, .kernel = "saxpy" },
                .divmod => .{ .kind = kind, .kernel = "divmod" },
                .tma => .{ .kind = kind, .kernel = "tmaTile" },
                .trap => .{ .kind = kind, .kernel = operand[0] },
                .expect => .{ .kind = kind, .kernel = operand[0], .words = try readWords(io, arena, operand[1]) },
                .contend => .{
                    .kind = kind,
                    .kernel = operand[0],
                    .grid = std.fmt.parseInt(u32, operand[1], 0) catch fatal("invalid grid {s}", .{operand[1]}),
                    .block = std.fmt.parseInt(u32, operand[2], 0) catch fatal("invalid block {s}", .{operand[2]}),
                    .words = try readWords(io, arena, operand[3]),
                },
            });
        }
    }
    const path = ptx_path orelse fatal(usage, .{});

    var device: ?Device = null;
    var skip_reason: []const u8 = "no CUDA driver";
    if (Cuda.Raw.load()) |cuda| {
        cuda.initDevice();
        const ptx = try Io.Dir.cwd().readFileAllocOptions(io, path, arena, .unlimited, .of(u8), 0);
        const sm = Sm.fromPtx(ptx) orelse fatal("{s} has no .target directive", .{path});
        const capability = cuda.computeCapability();
        if (sm.loadsOn(capability.major, capability.minor)) {
            device = .{ .cuda = cuda, .module = cuda.loadModule(ptx, path, link_paths.items) };
        } else {
            skip_reason = try std.fmt.allocPrint(arena, "needs {f}", .{sm});
        }
    } else |_| {}
    if (listen) return serve(io, arena, if (device) |*d| d else null, cases.items);

    if (device == null) std.debug.print("skipped ({s}): {s}\n", .{ skip_reason, path });

    var failures: usize = 0;
    for (cases.items) |case| {
        const status = run(if (device) |*d| d else null, case);
        std.debug.print("{s}: {s}\n", .{ case.kernel, switch (status) {
            .pass => "ok",
            .fail => "FAIL",
            .skip => "skipped",
        } });
        failures += @intFromBool(status == .fail);
    }
    if (failures != 0) std.process.exit(1);
}

fn readWords(io: Io, arena: std.mem.Allocator, operand: []const u8) ![]const u8 {
    const file = std.mem.cutPrefix(u8, operand, "@") orelse return operand;
    return Io.Dir.cwd().readFileAlloc(io, file, arena, .unlimited);
}

fn serve(io: Io, arena: std.mem.Allocator, device: ?*const Device, cases: []const Case) !void {
    var stdin_buffer: [4096]u8 = undefined;
    var stdout_buffer: [4096]u8 = undefined;
    var stdin_reader: Io.File.Reader = .initStreaming(.stdin(), io, &stdin_buffer);
    var stdout_writer: Io.File.Writer = .initStreaming(.stdout(), io, &stdout_buffer);
    var server: std.zig.Server = .{
        .in = &stdin_reader.interface,
        .out = &stdout_writer.interface,
    };
    try server.serveStringMessage(.zig_version, builtin.zig_version_string);

    while (true) {
        const hdr = try server.receiveMessage();
        switch (hdr.tag) {
            .exit => return std.process.exit(0),
            .query_test_metadata => {
                var string_bytes: std.ArrayList(u8) = .empty;
                try string_bytes.append(arena, 0); // Reserve 0 for null.
                const names = try arena.alloc(u32, cases.len);
                const expected_panic_msgs = try arena.alloc(u32, cases.len);
                for (cases, names, expected_panic_msgs) |case, *name, *expected_panic_msg| {
                    name.* = @intCast(string_bytes.items.len);
                    try string_bytes.appendSlice(arena, case.kernel);
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
                try server.serveTestResults(.{
                    .index = index,
                    .flags = .{
                        .status = run(device, cases[index]),
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

fn run(optional_device: ?*const Device, case: Case) Status {
    const device = optional_device orelse return .skip;
    const passed = switch (case.kind) {
        .saxpy => saxpy(device),
        .divmod => divmod(device),
        .tma => tma(device),
        .expect, .contend => compare(device, case),
        .trap => trap(device, case.kernel),
    };
    return if (passed) .pass else .fail;
}

fn saxpy(device: *const Device) bool {
    const n = 1000;
    var x: [n]f32 = undefined;
    var y: [n]f32 = undefined;
    var want: [n]f32 = undefined;
    const a: f32 = 2.5;
    const len: u32 = n;
    for (&x, &y, &want, 0..) |*xi, *yi, *wi, i| {
        xi.* = @as(f32, @floatFromInt(i)) * 0.5;
        yi.* = @floatFromInt(1000 - @as(i32, @intCast(i)));
        wi.* = a * xi.* + yi.*;
    }
    const dx = device.alloc(@sizeOf(@TypeOf(x)));
    const dy = device.alloc(@sizeOf(@TypeOf(y)));
    device.copyIn(dx, std.mem.asBytes(&x));
    device.copyIn(dy, std.mem.asBytes(&y));
    device.launchOrFail("saxpy", (n + 127) / 128, 128, &.{ &len, &a, &dx, &dy });
    device.copyOut(std.mem.asBytes(&y), dy);
    var failures: usize = 0;
    for (y, want, 0..) |got, expected, i| {
        if (got == expected) continue;
        if (failures < 5) std.debug.print("y[{d}] = {d}, want {d}\n", .{ i, got, expected });
        failures += 1;
    }
    return failures == 0;
}

fn divmod(device: *const Device) bool {
    const a: u32 = 47;
    const b: u32 = 5;
    var out: [3]u32 = undefined;
    const d = device.alloc(@sizeOf(@TypeOf(out)));
    device.launchOrFail("divmod", 1, 1, &.{ &d, &a, &b });
    device.copyOut(std.mem.asBytes(&out), d);
    if (std.mem.eql(u32, &out, &.{ 9, 2, 48 })) return true;
    std.debug.print("divmod = {d} {d} {d}\n", .{ out[0], out[1], out[2] });
    return false;
}

/// Loads the 16-row, 32-column box at column 32, row 16 of a 64 x 128 f32 tensor.
fn tma(device: *const Device) bool {
    const width = 128;
    const height = 64;
    const box_width = 32;
    const box_height = 16;
    const box_x = 32;
    const box_y = 16;
    var source: [height][width]f32 = undefined;
    for (&source, 0..) |*row, r| {
        for (row, 0..) |*element, c| element.* = @floatFromInt(r * 1000 + c);
    }
    var out: [box_height * box_width]f32 = undefined;
    const d = device.alloc(@sizeOf(@TypeOf(source)));
    const dout = device.alloc(@sizeOf(@TypeOf(out)));
    device.copyIn(d, std.mem.asBytes(&source));

    var map: Cuda.TensorMap = undefined;
    const dims = [2]u64{ width, height };
    const strides = [1]u64{width * @sizeOf(f32)};
    const box = [2]u32{ box_width, box_height };
    const element_strides = [2]u32{ 1, 1 };
    const encode = device.cuda.cuTensorMapEncodeTiled orelse fatal("the CUDA driver does not support tensor maps", .{});
    device.cuda.check(encode(
        &map,
        .float32,
        2,
        @ptrFromInt(d),
        &dims,
        &strides,
        &box,
        &element_strides,
        .none,
        .none,
        .none,
        .none,
    ), "cuTensorMapEncodeTiled");
    const x: i32 = box_x;
    const y: i32 = box_y;
    device.launchOrFail("tmaTile", 1, 128, &.{ &map, &dout, &x, &y });
    device.copyOut(std.mem.asBytes(&out), dout);

    var failures: usize = 0;
    for (0..box_height) |r| for (0..box_width) |c| {
        const want: f32 = @floatFromInt((box_y + r) * 1000 + (box_x + c));
        const got = out[r * box_width + c];
        if (got == want) continue;
        if (failures < 5) std.debug.print("out[{d}][{d}] = {d}, want {d}\n", .{ r, c, got, want });
        failures += 1;
    };
    return failures == 0;
}

fn compare(device: *const Device, case: Case) bool {
    const expected: Expected = .parse(case.words, if (case.kind == .expect) 0xdeadbeef else 0);
    var got = expected.initial;
    const d = device.alloc(@sizeOf(@TypeOf(got)));
    device.copyIn(d, std.mem.sliceAsBytes(&got));
    device.launchOrFail(case.kernel, case.grid, case.block, &.{&d});
    device.copyOut(std.mem.sliceAsBytes(got[0..expected.len]), d);
    var failures: usize = 0;
    for (got[0..expected.len], expected.want[0..expected.len], expected.check[0..expected.len], 0..) |g, w, check, i| {
        if (!check or g == w) continue;
        if (failures < 8) std.debug.print("{s}: out[{d}] = {d} (0x{x}), want {d} (0x{x})\n", .{ case.kernel, i, g, g, w, w });
        failures += 1;
    }
    return failures == 0;
}

fn trap(device: *const Device, kernel: [:0]const u8) bool {
    const d = device.alloc(word_count * @sizeOf(u32));
    const result = device.launch(kernel, 1, 1, &.{&d});
    const text = device.cuda.errorString(result);
    if (result != 0 and std.mem.find(u8, text, "launch failure") != null) return true;
    std.debug.print("{s} did not trap: {s}\n", .{ kernel, if (result == 0) "no error" else text });
    return false;
}
