const std = @import("std");
const cuda = @import("cuda");
const TerminalTableRenderer = @import("TerminalTableRenderer");
const cell = TerminalTableRenderer.cell;
const rms_norm = @import("rms_norm_kernel.zig");

const rows = 16384;
const hidden = rms_norm.hidden;
const epsilon = 1e-6;
/// The GB/s columns count x read and y written. They omit the weight row, which every row reads.
const bytes_per_launch = 2 * rows * hidden * @sizeOf(u16);

const Options = struct {
    warmup: u32 = 100,
    iterations: u32 = 20,
    repeat: u32 = 5,
    rounds: u32 = 3,
    seed: u64 = 1234,
};

const copy_index = for (rms_norm.kernels, 0..) |Kernel, index| {
    if (Kernel == rms_norm.Copy) break index;
} else unreachable;

const name_column = std.fmt.comptimePrint("{{s:<{d}}}", .{width: {
    var width = "Kernel".len;
    for (rms_norm.kernels) |Kernel| width = @max(width, Kernel.name.len);
    break :width width;
}});

const Runner = struct {
    functions: [rms_norm.kernels.len]cuda.Function,
    shared_bytes: u32,
    stream: cuda.Stream,

    fn launch(runner: *const Runner, comptime index: usize, problem: *const Problem) !void {
        const Kernel = rms_norm.kernels[index];
        try runner.functions[index].launch(.{
            .grid = .{ .x = rows / Kernel.warps },
            .block = .{ .x = Kernel.warps * 32 },
            .shared_memory_bytes = runner.shared_bytes,
            .stream = runner.stream,
        }, .{ problem.x, problem.weight, problem.y, @as(u32, rows), @as(f32, epsilon) });
    }

    /// Fills each element of `timings` with the mean time in milliseconds of one launch, over
    /// `iterations` launches.
    fn measure(
        runner: *const Runner,
        comptime index: usize,
        problem: *const Problem,
        iterations: u32,
        timings: []f64,
    ) !void {
        const start = try cuda.Event.create();
        defer start.destroy();
        const stop = try cuda.Event.create();
        defer stop.destroy();
        const launches: f64 = @floatFromInt(iterations);
        for (timings) |*milliseconds| {
            try start.record(runner.stream);
            for (0..iterations) |_| try runner.launch(index, problem);
            try stop.record(runner.stream);
            try stop.synchronize();
            milliseconds.* = @as(f64, try start.elapsed(stop)) / launches;
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const options = try parseOptions(init);
    const allocator = init.gpa;
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
    if ((try device.computeCapability()).major < 8) return error.RequiresSm80;
    var name: [256]u8 = undefined;
    try renderer.row(.title, "{s}: RMSNorm forward of {d} rows of {d} bf16 values, f32 math\n", .{
        try device.name(&name),
        rows,
        hidden,
    });

    const module = try cuda.Module.load(@embedFile("rms_norm_kernel.ptx"));
    defer module.unload();
    // The kernels use no shared memory. Each work group asks for the largest amount that a work
    // group may have, so that one work group fills an SM and `Warps/SM` equals `warps`.
    var runner: Runner = .{
        .functions = undefined,
        .shared_bytes = @intCast(try device.attribute(.max_shared_memory_per_block_optin)),
        .stream = try cuda.Stream.create(),
    };
    defer runner.stream.destroy();
    const resource_format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>9}  {d:>11}  {d:>8}";
    try renderer.row(.heading, name_column ++ "  {s:>9}  {s:>11}  {s:>8}", .{
        "Kernel",
        "Registers",
        "Local bytes",
        "Warps/SM",
    });
    inline for (rms_norm.kernels, &runner.functions) |Kernel, *function| {
        function.* = try module.function(Kernel.name);
        try function.setMaxDynamicSharedMemory(runner.shared_bytes);
        const groups = try function.maxActiveBlocksPerMultiprocessor(
            Kernel.warps * 32,
            runner.shared_bytes,
        );
        try renderer.row(.body, resource_format, .{
            Kernel.name,
            @as(u32, @intCast(try function.attribute(.num_regs))),
            @as(u32, @intCast(try function.attribute(.local_size_bytes))),
            groups * Kernel.warps,
        });
    }

    var problem = try Problem.init(allocator, options.seed);
    defer problem.deinit(allocator);
    var max_errors: [rms_norm.kernels.len]f64 = undefined;
    inline for (rms_norm.kernels, &max_errors, 0..) |Kernel, *max_error, index| {
        try problem.y.memsetAsync(0xff, runner.stream);
        try runner.launch(index, &problem);
        try runner.stream.synchronize();
        max_error.* = try problem.check(Kernel == rms_norm.RmsNorm);
    }

    try renderer.row(.title, "\n{d} warmup launches, {d} rounds of {d} samples of {d} launches", .{
        options.warmup,
        options.rounds,
        options.repeat,
        options.iterations,
    });
    try renderer.row(.heading, name_column ++ "  {s:>9}  {s:>7}  {s:>9}  {s:>8}  {s:>9}", .{
        "Kernel",
        "Median ms",
        "GB/s",
        "Best GB/s",
        "Of copy",
        "Max error",
    });
    inline for (0..rms_norm.kernels.len) |index| {
        for (0..options.warmup) |_| try runner.launch(index, &problem);
    }
    const samples = options.rounds * options.repeat;
    const timings = try allocator.alloc(f64, (rms_norm.kernels.len + 1) * samples);
    defer allocator.free(timings);
    const scratch = timings[rms_norm.kernels.len * samples ..];
    renderer.keep();
    // The kernels take turns in each round, so that a slow drift of the GPU speed affects each of
    // them alike.
    for (0..options.rounds) |round| {
        inline for (0..rms_norm.kernels.len) |index| {
            const first = index * samples + round * options.repeat;
            const round_timings = timings[first..][0..options.repeat];
            try runner.measure(index, &problem, options.iterations, round_timings);
        }
        if (!renderer.terminal and round + 1 < options.rounds) continue;
        const measured = (round + 1) * options.repeat;
        const copy_time = median(timings[copy_index * samples ..][0..measured], scratch);
        try renderer.rewind();
        inline for (rms_norm.kernels, 0..) |Kernel, index| {
            const kernel_timings = timings[index * samples ..][0..measured];
            const time = median(kernel_timings, scratch);
            try renderer.row(
                .body,
                cell.cyan ++ name_column ++ cell.normal ++ "  {d:>9.4}  " ++ cell.bold ++
                    "{d:>7.1}" ++ cell.normal ++ "  {d:>9.1}  {d:>7.1}%  " ++ cell.dim ++
                    "{e:>9.3}" ++ cell.normal,
                .{
                    Kernel.name,
                    time,
                    bytes_per_launch / (time * 1e6),
                    bytes_per_launch / (std.mem.min(f64, kernel_timings) * 1e6),
                    100 * copy_time / time,
                    max_errors[index],
                },
            );
        }
    }
}

/// Assumes that `values` is not empty and that `scratch` is at least as long. Returns the median
/// of `values`. The function sorts a copy of `values` in `scratch`.
fn median(values: []const f64, scratch: []f64) f64 {
    const sorted = scratch[0..values.len];
    @memcpy(sorted, values);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const middle = sorted.len / 2;
    return if (sorted.len % 2 == 0) (sorted[middle - 1] + sorted[middle]) / 2 else sorted[middle];
}

const Problem = struct {
    x_host: []u16,
    weight_host: []u16,
    y_host: []u16,
    x: cuda.Buffer(u16),
    weight: cuda.Buffer(u16),
    y: cuda.Buffer(u16),

    /// The rows of x differ in scale, so that a kernel that takes another row's sum fails `check`.
    /// The weights differ from 1, so that a kernel that skips them fails it too.
    fn init(allocator: std.mem.Allocator, seed: u64) !Problem {
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        const x_host = try allocator.alloc(u16, rows * hidden);
        errdefer allocator.free(x_host);
        const weight_host = try allocator.alloc(u16, hidden);
        errdefer allocator.free(weight_host);
        const y_host = try allocator.alloc(u16, rows * hidden);
        errdefer allocator.free(y_host);
        for (x_host, 0..) |*element, index| {
            const row_scale: f32 = @floatFromInt(1 + index / hidden % 7);
            element.* = toBf16(random.floatNorm(f32) * row_scale);
        }
        for (weight_host) |*element| element.* = toBf16(1 + 0.25 * random.floatNorm(f32));
        const x = try cuda.dupe(u16, x_host);
        errdefer x.free();
        const weight = try cuda.dupe(u16, weight_host);
        errdefer weight.free();
        return .{
            .x_host = x_host,
            .weight_host = weight_host,
            .y_host = y_host,
            .x = x,
            .weight = weight,
            .y = try cuda.alloc(u16, rows * hidden),
        };
    }

    fn deinit(problem: *Problem, allocator: std.mem.Allocator) void {
        problem.x.free();
        problem.weight.free();
        problem.y.free();
        allocator.free(problem.x_host);
        allocator.free(problem.weight_host);
        allocator.free(problem.y_host);
    }

    /// Copies y to the host and compares every value with the f64 RMSNorm of x, or with x for the
    /// copy kernel. Returns the largest absolute error. A value outside the tolerance fails it.
    fn check(problem: *Problem, normalized: bool) !f64 {
        try problem.y.copyTo(problem.y_host);
        var max_error: f64 = 0;
        for (0..rows) |row| {
            const x_row = problem.x_host[row * hidden ..][0..hidden];
            const y_row = problem.y_host[row * hidden ..][0..hidden];
            var squares: f64 = 0;
            for (x_row) |bits| squares += square(fromBf16(bits));
            const scale = 1 / @sqrt(squares / hidden + epsilon);
            for (x_row, y_row, problem.weight_host, 0..) |x_bits, y_bits, w_bits, column| {
                const x: f64 = fromBf16(x_bits);
                const expected = if (normalized) x * scale * fromBf16(w_bits) else x;
                const actual: f64 = fromBf16(y_bits);
                const difference = @abs(actual - expected);
                max_error = @max(max_error, difference);
                // Rounding to bf16 changes a value by at most 2^-8 of its size, because bf16 has
                // 8 significant bits. The factor 1.01 leaves room for the f32 arithmetic.
                if (!std.math.isFinite(actual) or difference > 0x1p-8 * 1.01 * @abs(expected)) {
                    std.log.err("Mismatch at row {d}, column {d}: actual={d}, expected={d}", .{
                        row, column, actual, expected,
                    });
                    return error.IncorrectResult;
                }
            }
        }
        return max_error;
    }
};

fn square(value: f64) f64 {
    return value * value;
}

fn parseOptions(init: std.process.Init) !Options {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var options: Options = .{};
    for (args[1..]) |arg| {
        const equals = std.mem.findScalar(u8, arg, '=') orelse
            return error.ExpectedNameEqualsNumber;
        inline for (comptime std.meta.fieldNames(Options)) |field_name| {
            if (std.mem.eql(u8, arg[0..equals], "--" ++ field_name)) {
                const Field = @FieldType(Options, field_name);
                @field(options, field_name) = try std.fmt.parseInt(Field, arg[equals + 1 ..], 10);
                break;
            }
        } else return error.UnknownOption;
    }
    if (options.iterations == 0 or options.repeat == 0 or options.rounds == 0 or
        options.rounds > 16) return error.InvalidBenchmarkCount;
    return options;
}

fn toBf16(number: f32) u16 {
    const bits: u32 = @bitCast(number);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

fn fromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
