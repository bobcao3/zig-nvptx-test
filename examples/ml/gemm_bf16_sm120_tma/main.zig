const std = @import("std");
const cuda = @import("cuda");
const TerminalTableRenderer = @import("TerminalTableRenderer");
const cell = TerminalTableRenderer.cell;
const Blas = @import("Blas.zig");
const gemm = @import("gemm_kernel.zig");

const name_column = std.fmt.comptimePrint("{{s:<{d}}}", .{width: {
    var width = "Kernel".len;
    for (gemm.kernels) |Kernel| width = @max(width, Kernel.name.len);
    break :width width;
}});

const Shape = struct { m: u32, n: u32, k: u32 };
const Options = struct {
    warmup: u32 = 20,
    iterations: u32 = 50,
    repeat: u32 = 7,
    seed: u64 = 1234,
    size: u32 = 4096,
};

pub fn main(init: std.process.Init) !void {
    const options = try parseOptions(init);
    const stdout = std.Io.File.stdout();
    var buffer: [1024]u8 = undefined;
    var stdout_writer = stdout.writer(init.io, &buffer);
    var renderer: TerminalTableRenderer = try .init(
        init.io,
        stdout,
        &stdout_writer.interface,
        init.environ_map,
    );
    const device = try cuda.init(0);
    defer device.deinit();
    const capability = try device.computeCapability();
    if (capability.major != 12) return error.RequiresSm12x;
    var name: [256]u8 = undefined;
    const title = "{s}: C = A[M,K] B[N,K]^T with bf16 inputs and f32 accumulation and output\n";
    try renderer.row(.title, title, .{try device.name(&name)});
    const module = try cuda.Module.load(@embedFile("gemm_kernel.ptx"));
    defer module.unload();
    const stream = try cuda.Stream.create();
    defer stream.destroy();
    var blas = try Blas.open(stream);
    defer blas.close();
    var functions: [gemm.kernels.len]cuda.Function = undefined;
    try renderer.row(.heading, name_column ++ "  {s:>9}  {s:>12}  {s:>14}", .{
        "Kernel",
        "Registers",
        "Shared bytes",
        "Work groups/SM",
    });
    inline for (gemm.kernels, &functions) |Kernel, *function| {
        function.* = try module.function(Kernel.name);
        try function.setMaxDynamicSharedMemory(Kernel.dynamic_shared_bytes);
        const static_shared: u32 = @intCast(try function.attribute(.shared_size_bytes));
        const shared = Kernel.dynamic_shared_bytes;
        const format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>9}  {d:>12}  {d:>14}";
        try renderer.row(.body, format, .{
            Kernel.name,
            @as(u32, @intCast(try function.attribute(.num_regs))),
            static_shared + shared,
            try function.maxActiveBlocksPerMultiprocessor(Kernel.threads, shared),
        });
    }
    try renderer.row(.title, "\nChecks against the f32 cuBLAS result", .{});
    try renderer.row(.heading, name_column ++ "  {s:>5}  {s:>5}  {s:>5}  {s:>9}  {s:>9}", .{
        "Kernel", "M", "N", "K", "Max error", "RMS error",
    });
    inline for (gemm.kernels, functions) |Kernel, function| {
        for ([_]Shape{
            .{ .m = Kernel.block_m, .n = Kernel.block_n * 2, .k = Kernel.block_k },
            .{ .m = Kernel.block_m * 2, .n = Kernel.block_n, .k = Kernel.block_k * 2 },
            .{ .m = Kernel.block_m, .n = Kernel.block_n * 2, .k = Kernel.block_k * 3 },
            .{ .m = Kernel.block_m * 2, .n = Kernel.block_n, .k = Kernel.block_k * 5 },
            .{
                .m = Kernel.block_m * (Kernel.group_m + 1),
                .n = Kernel.block_n * 3,
                .k = Kernel.block_k * 4,
            },
        }) |shape| {
            try runCase(Kernel, init.gpa, function, stream, blas, shape, options, false, &renderer);
        }
    }
    const size = options.size;
    try renderer.row(.title, "\n{d}x{d}x{d}: {d} warmup launches, {d} samples of {d} launches", .{
        size,
        size,
        size,
        options.warmup,
        options.repeat,
        options.iterations,
    });
    try renderer.row(.heading, name_column ++ "  {s:>9}  {s:>6}  {s:>9}  {s:>7}  {s:>9}", .{
        "Kernel",
        "Median ms",
        "TF/s",
        "Best TF/s",
        "P90 ms",
        "Max error",
    });
    const shape: Shape = .{ .m = size, .n = size, .k = size };
    inline for (gemm.kernels, functions) |Kernel, function| {
        try runCase(Kernel, init.gpa, function, stream, blas, shape, options, true, &renderer);
    }
}

fn parseOptions(init: std.process.Init) !Options {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var options: Options = .{};
    for (args[1..]) |arg| {
        const equals = std.mem.findScalar(u8, arg, '=') orelse return error.ExpectedNameEqualsNumber;
        const key = arg[0..equals];
        const value = arg[equals + 1 ..];
        if (std.mem.eql(u8, key, "--warmup")) {
            options.warmup = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, key, "--iterations")) {
            options.iterations = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, key, "--repeat")) {
            options.repeat = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, key, "--seed")) {
            options.seed = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, key, "--size")) {
            options.size = try std.fmt.parseInt(u32, value, 10);
        } else return error.UnknownOption;
    }
    if (options.warmup == 0 or options.warmup > 10000 or options.iterations == 0 or options.iterations > 10000 or options.repeat == 0 or options.repeat > 1000) return error.InvalidBenchmarkCount;
    if (options.size == 0 or options.size > 8192) return error.InvalidBenchmarkSize;
    inline for (gemm.kernels) |Kernel| {
        if (options.size % Kernel.block_m != 0 or options.size % Kernel.block_n != 0 or
            options.size % Kernel.block_k != 0) return error.InvalidBenchmarkSize;
    }
    return options;
}

/// Checks the kernel on `shape` against cuBLAS. With `benchmark`, the function also times the
/// kernel and writes a row of the benchmark table, and otherwise it writes a row of the checks.
fn runCase(
    comptime Kernel: type,
    allocator: std.mem.Allocator,
    kernel: cuda.Function,
    stream: cuda.Stream,
    blas: Blas,
    shape: Shape,
    options: Options,
    benchmark: bool,
    renderer: *TerminalTableRenderer,
) !void {
    const a_host = try allocator.alloc(u16, @as(usize, shape.m) * shape.k);
    defer allocator.free(a_host);
    const b_host = try allocator.alloc(u16, @as(usize, shape.n) * shape.k);
    defer allocator.free(b_host);
    var prng: std.Random.DefaultPrng = .init(options.seed);
    const random = prng.random();
    for (a_host) |*element| element.* = toBf16(random.floatNorm(f32));
    const weight_std = 1.0 / @sqrt(@as(f32, @floatFromInt(shape.k)));
    for (b_host) |*element| element.* = toBf16(random.floatNorm(f32) * weight_std);
    const a = try cuda.dupe(u16, a_host);
    defer a.free();
    const b = try cuda.dupe(u16, b_host);
    defer b.free();
    const c = try cuda.alloc(f32, @as(usize, shape.m) * shape.n);
    defer c.free();
    if (c.address % 16 != 0 or (@as(u64, shape.n) * @sizeOf(f32)) % 16 != 0)
        return error.MisalignedOutput;
    const reference = try cuda.alloc(f32, c.len);
    defer reference.free();
    const a_map = try tensorMap(Kernel, a.address, shape.m, shape.k, Kernel.block_m);
    const b_map = try tensorMap(Kernel, b.address, shape.n, shape.k, Kernel.block_n);
    const launch: cuda.LaunchConfig = .{
        .grid = .{ .x = shape.n / Kernel.block_n, .y = shape.m / Kernel.block_m },
        .block = .{ .x = Kernel.threads },
        .shared_memory_bytes = Kernel.dynamic_shared_bytes,
        .stream = stream,
    };
    const args = .{ a_map, b_map, c, shape.m, shape.n, shape.k };
    try c.memsetAsync(0xff, stream);
    try kernel.launch(launch, args);
    const a_float = try allocator.alloc(f32, a_host.len);
    defer allocator.free(a_float);
    const b_float = try allocator.alloc(f32, b_host.len);
    defer allocator.free(b_float);
    for (a_float, a_host) |*dst, src| dst.* = fromBf16(src);
    for (b_float, b_host) |*dst, src| dst.* = fromBf16(src);
    const a_reference = try cuda.dupe(f32, a_float);
    defer a_reference.free();
    const b_reference = try cuda.dupe(f32, b_float);
    defer b_reference.free();
    try blas.reference(a_reference, b_reference, reference, shape.m, shape.n, shape.k);
    try stream.synchronize();
    const computed = try allocator.alloc(f32, c.len);
    defer allocator.free(computed);
    const expected = try allocator.alloc(f32, c.len);
    defer allocator.free(expected);
    try c.copyTo(computed);
    try reference.copyTo(expected);
    const accuracy = try compare(computed, expected, shape);
    if (!benchmark) {
        const format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>5}  {d:>5}  {d:>5}  " ++
            "{e:>9.3}  {e:>9.3}";
        return renderer.row(.body, format, .{
            Kernel.name,
            shape.m,
            shape.n,
            shape.k,
            accuracy.max_abs,
            accuracy.rms,
        });
    }
    renderer.keep();
    if (renderer.terminal) {
        const measuring = cell.dim ++ cell.italic ++ "measuring" ++ cell.normal;
        const placeholder = cell.cyan ++ name_column ++ cell.normal ++ "  " ++ measuring;
        try renderer.row(.body, placeholder, .{Kernel.name});
    }
    for (0..options.warmup) |_| try kernel.launch(launch, args);
    try stream.synchronize();
    const start = try cuda.Event.create();
    defer start.destroy();
    const stop = try cuda.Event.create();
    defer stop.destroy();
    const timings = try allocator.alloc(f64, options.repeat);
    defer allocator.free(timings);
    for (timings) |*milliseconds| {
        try start.record(stream);
        for (0..options.iterations) |_| try kernel.launch(launch, args);
        try stop.record(stream);
        try stop.synchronize();
        milliseconds.* = @as(f64, try start.elapsed(stop)) / @as(f64, @floatFromInt(options.iterations));
        if (!std.math.isFinite(milliseconds.*) or milliseconds.* <= 0) return error.InvalidDeviceTiming;
    }
    try c.copyTo(computed);
    _ = try compare(computed, expected, shape);
    std.mem.sort(f64, timings, {}, std.sort.asc(f64));
    const middle = timings.len / 2;
    const median = if (timings.len % 2 == 0) (timings[middle - 1] + timings[middle]) / 2 else timings[middle];
    const operations = 2.0 * @as(f64, @floatFromInt(shape.m)) * @as(f64, @floatFromInt(shape.n)) * @as(f64, @floatFromInt(shape.k));
    try renderer.rewind();
    const format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>9.4}  " ++ cell.bold ++
        "{d:>6.2}" ++ cell.normal ++ "  {d:>9.2}  {d:>7.4}  " ++ cell.dim ++ "{e:>9.3}" ++
        cell.normal;
    try renderer.row(.body, format, .{
        Kernel.name,
        median,
        operations / (median * 1e9),
        operations / (timings[0] * 1e9),
        timings[(timings.len * 9 + 9) / 10 - 1],
        accuracy.max_abs,
    });
}

fn compare(computed: []const f32, expected: []const f32, shape: Shape) !struct { max_abs: f64, rms: f64 } {
    var max_error: f64 = 0;
    var squared_error: f64 = 0;
    var failures: usize = 0;
    for (computed, expected, 0..) |actual, reference, index| {
        const difference = @abs(@as(f64, actual) - reference);
        max_error = @max(max_error, difference);
        squared_error += difference * difference;
        if (!std.math.isFinite(actual) or !std.math.isFinite(reference) or difference > 1e-4 + 1e-4 * @abs(@as(f64, reference))) {
            if (failures < 4) std.log.err("Mismatch [{d},{d}]: actual={d}, reference={d}", .{ index / shape.n, index % shape.n, actual, reference });
            failures += 1;
        }
    }
    if (failures != 0) {
        std.log.err("{d}/{d} outputs failed correctness", .{ failures, computed.len });
        return error.IncorrectGemm;
    }
    return .{
        .max_abs = max_error,
        .rms = @sqrt(squared_error / @as(f64, @floatFromInt(computed.len))),
    };
}

fn toBf16(number: f32) u16 {
    const bits: u32 = @bitCast(number);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

fn fromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn tensorMap(comptime Kernel: type, address: cuda.DevicePointer, rows: u32, columns: u32, tile_rows: u32) !cuda.TensorMap {
    return cuda.TensorMap.encodeTiled(.{
        .data_type = .bfloat16,
        .address = address,
        .dims = &.{ columns, rows },
        .strides = &.{@as(u64, columns) * @sizeOf(u16)},
        .box = &.{ Kernel.block_k, tile_rows },
        .swizzle = if (Kernel.swizzle_bytes == 64) .@"64b" else .@"128b",
    });
}
