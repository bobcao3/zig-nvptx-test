const std = @import("std");
const cuda = @import("cuda");
const TerminalTableRenderer = @import("TerminalTableRenderer");
const cell = TerminalTableRenderer.cell;
const Blas = @import("Blas.zig");
const gemm = @import("gemm_kernel.zig");
const kernel_candidates = [_]Candidate{ .kernel, .pipelined };

const Shape = struct { m: u32, n: u32, k: u32 };
const Options = struct {
    warmup: u32 = 20,
    iterations: u32 = 50,
    repeat: u32 = 7,
    rounds: u32 = 3,
    seed: u64 = 1234,
    size: u32 = 4096,
    cublas: u32 = 1,
};

/// The implementations that the benchmark times: the kernel and, with `--cublas`, cuBLAS.
const Candidate = enum {
    kernel,
    pipelined,
    cublas,

    fn name(candidate: Candidate) []const u8 {
        return switch (candidate) {
            .kernel => gemm.Kernel.name,
            .pipelined => gemm.PipelinedKernel.name,
            .cublas => "cublasGemmEx",
        };
    }
};

const name_column = std.fmt.comptimePrint("{{s:<{d}}}", .{@max(gemm.Kernel.name.len, gemm.PipelinedKernel.name.len, "Candidate".len)});

const Runner = struct {
    functions: [gemm.kernels.len]cuda.Function,
    blas: Blas,
    stream: cuda.Stream,

    fn launch(runner: *const Runner, candidate: Candidate, problem: *const Problem) !void {
        const shape = problem.shape;
        switch (candidate) {
            .kernel, .pipelined => {
                inline for (gemm.kernels, runner.functions, kernel_candidates) |Kernel, function, kernel_candidate| {
                    if (candidate == kernel_candidate) {
                        try function.launch(.{
                            .grid = .{ .x = shape.n / Kernel.block_n, .y = shape.m / Kernel.block_m },
                            .block = .{ .x = Kernel.threads },
                            .shared_memory_bytes = Kernel.dynamic_shared_bytes,
                            .stream = runner.stream,
                        }, .{ problem.a, problem.b, problem.c, shape.m, shape.n, shape.k });
                        return;
                    }
                }
                unreachable;
            },
            .cublas => try runner.blas.gemmBf16(
                problem.a,
                problem.b,
                problem.c,
                shape.m,
                shape.n,
                shape.k,
            ),
        }
    }

    /// Fills each element of `timings` with the mean time in milliseconds of one launch, over
    /// `options.iterations` launches.
    fn measure(
        runner: *const Runner,
        candidate: Candidate,
        problem: *const Problem,
        options: Options,
        timings: []f64,
    ) !void {
        for (0..options.warmup) |_| try runner.launch(candidate, problem);
        try runner.stream.synchronize();
        const start = try cuda.Event.create();
        defer start.destroy();
        const stop = try cuda.Event.create();
        defer stop.destroy();
        const iterations: f64 = @floatFromInt(options.iterations);
        for (timings) |*milliseconds| {
            try start.record(runner.stream);
            for (0..options.iterations) |_| try runner.launch(candidate, problem);
            try stop.record(runner.stream);
            try stop.synchronize();
            milliseconds.* = @as(f64, try start.elapsed(stop)) / iterations;
            if (!std.math.isFinite(milliseconds.*) or milliseconds.* <= 0) {
                return error.InvalidDeviceTiming;
            }
        }
    }

    /// Launches the candidate once on an output full of NaN, compares the output with the reference
    /// and writes a row of the checks. A value that the candidate does not write fails the check.
    fn validate(
        runner: *const Runner,
        renderer: *TerminalTableRenderer,
        candidate: Candidate,
        problem: *Problem,
    ) !void {
        try problem.c.memsetAsync(0xff, runner.stream);
        try runner.launch(candidate, problem);
        try runner.stream.synchronize();
        const accuracy = try problem.check();
        const format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>5}  {d:>5}  {d:>5}  " ++
            "{e:>9.3}  {e:>9.3}";
        try renderer.row(.body, format, .{
            candidate.name(),
            problem.shape.m,
            problem.shape.n,
            problem.shape.k,
            accuracy.max_abs,
            accuracy.rms,
        });
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
    const capability = try device.computeCapability();
    if (capability.major < 8) return error.RequiresSm80;
    var name: [256]u8 = undefined;
    const title = "{s}: C = A[M,K] B[N,K]^T with bf16 inputs and f32 accumulation and output\n";
    try renderer.row(.title, title, .{try device.name(&name)});
    const module = try cuda.Module.load(@embedFile("gemm_kernel.ptx"));
    defer module.unload();
    const stream = try cuda.Stream.create();
    defer stream.destroy();
    var reference_blas = try Blas.open(stream, .pedantic);
    defer reference_blas.close();
    var functions: [gemm.kernels.len]cuda.Function = undefined;
    try renderer.row(.heading, name_column ++ "  {s:>9}  {s:>12}  {s:>14}", .{
        "Kernel",
        "Registers",
        "Shared bytes",
        "Work groups/SM",
    });
    const resource_format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>9}  {d:>12}  {d:>14}";
    inline for (gemm.kernels, &functions) |Kernel, *function| {
        function.* = try module.function(Kernel.name);
        try function.setMaxDynamicSharedMemory(Kernel.dynamic_shared_bytes);
        const static_shared: u32 = @intCast(try function.attribute(.shared_size_bytes));
        const shared = Kernel.dynamic_shared_bytes;
        try renderer.row(.body, resource_format, .{
            Kernel.name,
            @as(u32, @intCast(try function.attribute(.num_regs))),
            static_shared + shared,
            try function.maxActiveBlocksPerMultiprocessor(Kernel.threads, shared),
        });
    }
    var runner: Runner = .{
        .functions = functions,
        .blas = try Blas.open(stream, .default),
        .stream = stream,
    };
    defer runner.blas.close();
    const size = options.size;
    const shape: Shape = .{ .m = size, .n = size, .k = size };
    var problem = try Problem.init(allocator, reference_blas, shape, options.seed);
    defer problem.deinit(allocator);
    const with_cublas: []const Candidate = &.{ .kernel, .pipelined, .cublas };
    const candidates = if (options.cublas != 0) with_cublas else &kernel_candidates;

    try renderer.row(.title, "\nChecks against the f32 cuBLAS result", .{});
    try renderer.row(.heading, name_column ++ "  {s:>5}  {s:>5}  {s:>5}  {s:>9}  {s:>9}", .{
        "Candidate", "M", "N", "K", "Max error", "RMS error",
    });
    for (candidates) |candidate| try runner.validate(&renderer, candidate, &problem);
    inline for (gemm.kernels, kernel_candidates) |Kernel, candidate| {
        for ([_]Shape{
            .{ .m = Kernel.block_m, .n = Kernel.block_n * 2, .k = Kernel.block_k },
            .{ .m = Kernel.block_m * 2, .n = Kernel.block_n, .k = Kernel.block_k * 2 },
            .{ .m = Kernel.block_m, .n = Kernel.block_n * 2, .k = Kernel.block_k * 3 },
            .{ .m = Kernel.block_m * 2, .n = Kernel.block_n, .k = Kernel.block_k * 5 },
            .{
                .m = Kernel.block_m * (Kernel.group_m + 1),
                .n = Kernel.block_n * 3,
                .k = Kernel.block_k * 9,
            },
        }) |small_shape| {
            var small = try Problem.init(allocator, reference_blas, small_shape, options.seed);
            defer small.deinit(allocator);
            try runner.validate(&renderer, candidate, &small);
        }
    }

    try renderer.row(.title, "\n{d}x{d}x{d}: {d} warmup launches before each round, " ++
        "{d} rounds of {d} samples of {d} launches", .{
        size,
        size,
        size,
        options.warmup,
        options.rounds,
        options.repeat,
        options.iterations,
    });
    try renderer.row(.heading, name_column ++ "  {s:>9}  {s:>6}  {s:>9}  {s}", .{
        "Candidate",
        "Median ms",
        "TF/s",
        "Best TF/s",
        "Round TF/s",
    });
    // The candidates take turns in each round, so that a slow drift of the GPU speed affects each
    // of them alike.
    const samples = options.rounds * options.repeat;
    const timings = try allocator.alloc(f64, (candidates.len + 1) * samples);
    defer allocator.free(timings);
    const scratch = timings[candidates.len * samples ..];
    renderer.keep();
    for (0..options.rounds) |round| {
        for (candidates, 0..) |candidate, slot| {
            const first = slot * samples + round * options.repeat;
            const round_timings = timings[first..][0..options.repeat];
            try runner.measure(candidate, &problem, options, round_timings);
            _ = try problem.check();
        }
        if (!renderer.terminal and round + 1 < options.rounds) continue;
        const measured = (round + 1) * options.repeat;
        var times: [kernel_candidates.len + 1]f64 = undefined;
        for (times[0..candidates.len], 0..) |*time, slot| {
            time.* = median(timings[slot * samples ..][0..measured], scratch);
        }
        const fastest = std.mem.findMin(f64, times[0..candidates.len]);
        try renderer.rewind();
        for (candidates, 0..) |candidate, slot| {
            const look: TerminalTableRenderer.Look = if (slot == fastest) .highlight else .body;
            const candidate_timings = timings[slot * samples ..][0..measured];
            try report(
                &renderer,
                look,
                candidate.name(),
                shape,
                candidate_timings,
                options.repeat,
                scratch,
            );
        }
    }
}

/// Writes the table row of one candidate with the median throughput of each round so far.
fn report(
    renderer: *TerminalTableRenderer,
    look: TerminalTableRenderer.Look,
    name: []const u8,
    shape: Shape,
    timings: []const f64,
    repeat: u32,
    scratch: []f64,
) !void {
    const m: f64 = @floatFromInt(shape.m);
    const n: f64 = @floatFromInt(shape.n);
    const k: f64 = @floatFromInt(shape.k);
    const operations = 2.0 * m * n * k;
    var text: [16 * 8]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&text);
    var first: usize = 0;
    while (first < timings.len) : (first += repeat) {
        const round_median = median(timings[first..][0..repeat], scratch);
        try writer.print(" {d:>6.2}", .{operations / (round_median * 1e9)});
    }
    const overall = median(timings, scratch);
    const format = cell.cyan ++ name_column ++ cell.normal ++ "  {d:>9.4}  " ++ cell.bold ++
        "{d:>6.2}" ++ cell.normal ++ "  {d:>9.2}" ++ cell.dim ++ "{s}" ++ cell.normal;
    try renderer.row(look, format, .{
        name,
        overall,
        operations / (overall * 1e9),
        operations / (std.mem.min(f64, timings) * 1e9),
        writer.buffered(),
    });
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

fn parseOptions(init: std.process.Init) !Options {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var options: Options = .{};
    for (args[1..]) |arg| {
        const equals = std.mem.findScalar(u8, arg, '=') orelse
            return error.ExpectedNameEqualsNumber;
        const key = arg[0..equals];
        const value = arg[equals + 1 ..];
        inline for (comptime std.meta.fieldNames(Options)) |field_name| {
            if (std.mem.eql(u8, key, "--" ++ field_name)) {
                const Field = @FieldType(Options, field_name);
                @field(options, field_name) = try std.fmt.parseInt(Field, value, 10);
                break;
            }
        } else return error.UnknownOption;
    }
    if (options.warmup == 0 or options.warmup > 10000 or
        options.iterations == 0 or options.iterations > 10000 or
        options.repeat == 0 or options.repeat > 1000 or
        options.rounds == 0 or options.rounds > 16)
        return error.InvalidBenchmarkCount;
    if (options.size == 0 or options.size > 8192) return error.InvalidBenchmarkSize;
    inline for (gemm.kernels) |Kernel| {
        if (options.size % Kernel.block_m != 0 or options.size % Kernel.block_n != 0 or
            options.size % Kernel.block_k != 0) return error.InvalidBenchmarkSize;
    }
    return options;
}

/// A problem holds random bf16 inputs in device memory, an output buffer and the f32 cuBLAS result
/// for the inputs, which `check` takes as the reference.
const Problem = struct {
    shape: Shape,
    a: cuda.Buffer(u16),
    b: cuda.Buffer(u16),
    c: cuda.Buffer(f32),
    expected: []f32,
    computed: []f32,

    fn init(allocator: std.mem.Allocator, blas: Blas, shape: Shape, seed: u64) !Problem {
        const a_host = try allocator.alloc(u16, @as(usize, shape.m) * shape.k);
        defer allocator.free(a_host);
        const b_host = try allocator.alloc(u16, @as(usize, shape.n) * shape.k);
        defer allocator.free(b_host);
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        for (a_host) |*element| element.* = toBf16(random.floatNorm(f32));
        const weight_std = 1.0 / @sqrt(@as(f32, @floatFromInt(shape.k)));
        for (b_host) |*element| element.* = toBf16(random.floatNorm(f32) * weight_std);
        const a = try cuda.dupe(u16, a_host);
        errdefer a.free();
        const b = try cuda.dupe(u16, b_host);
        errdefer b.free();
        const c = try cuda.alloc(f32, @as(usize, shape.m) * shape.n);
        errdefer c.free();
        if (c.address % 16 != 0 or (@as(u64, shape.n) * @sizeOf(f32)) % 16 != 0) {
            return error.MisalignedOutput;
        }
        const expected = try allocator.alloc(f32, c.len);
        errdefer allocator.free(expected);
        const computed = try allocator.alloc(f32, c.len);
        errdefer allocator.free(computed);
        {
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
            const reference = try cuda.alloc(f32, c.len);
            defer reference.free();
            try cuda.synchronize();
            try blas.reference(a_reference, b_reference, reference, shape.m, shape.n, shape.k);
            try cuda.synchronize();
            try reference.copyTo(expected);
        }
        return .{
            .shape = shape,
            .a = a,
            .b = b,
            .c = c,
            .expected = expected,
            .computed = computed,
        };
    }

    fn deinit(problem: *Problem, allocator: std.mem.Allocator) void {
        problem.a.free();
        problem.b.free();
        problem.c.free();
        allocator.free(problem.expected);
        allocator.free(problem.computed);
    }

    /// Copies the output to the host and compares it with the reference.
    fn check(problem: *Problem) !Accuracy {
        try problem.c.copyTo(problem.computed);
        return compare(problem.computed, problem.expected, problem.shape);
    }
};

const Accuracy = struct { max_abs: f64, rms: f64 };

fn compare(computed: []const f32, expected: []const f32, shape: Shape) !Accuracy {
    var max_error: f64 = 0;
    var squared_error: f64 = 0;
    var failures: usize = 0;
    for (computed, expected, 0..) |actual, reference, index| {
        const difference = @abs(@as(f64, actual) - reference);
        max_error = @max(max_error, difference);
        squared_error += difference * difference;
        const tolerance = 1e-4 + 1e-4 * @abs(@as(f64, reference));
        if (!std.math.isFinite(actual) or !std.math.isFinite(reference) or difference > tolerance) {
            if (failures < 4) std.log.err("Mismatch [{d},{d}]: actual={d}, reference={d}", .{
                index / shape.n, index % shape.n, actual, reference,
            });
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
