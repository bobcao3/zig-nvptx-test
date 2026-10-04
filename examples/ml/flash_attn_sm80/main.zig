const std = @import("std");
const cuda = @import("cuda");
const attention = @import("attention_kernel.zig");
const TerminalTableRenderer = @import("TerminalTableRenderer");
const cell = TerminalTableRenderer.cell;

const Options = struct {
    warmup: u32 = 3,
    iterations: u32 = 10,
    repeat: u32 = 5,
    rounds: u32 = 3,
    seed: u64 = 1234,
    batch: u32 = 4,
    heads: u32 = 16,
    /// The benchmarks use this sequence length, or 1024 and 4096 for the value 0.
    seq: u32 = 0,
    /// The value 0 skips the benchmarks.
    benchmark: u32 = 1,
    /// The value 0 skips the checks of the small problems.
    check: u32 = 1,
};

const Shape = struct {
    batch: u32,
    heads: u32,
    seq: u32,
    head_dim: u32,

    fn elements(shape: Shape) usize {
        return @as(usize, shape.batch) * shape.heads * shape.seq * shape.head_dim;
    }

    /// Returns the floating-point operations of one launch. Causal attention counts half.
    fn operations(shape: Shape, causal: bool) f64 {
        const seq: f64 = @floatFromInt(shape.seq);
        const heads: f64 = @floatFromInt(@as(u64, shape.batch) * shape.heads);
        const full = 4 * heads * seq * seq * @as(f64, @floatFromInt(shape.head_dim));
        return if (causal) full / 2 else full;
    }
};

const Runner = struct {
    functions: [attention.kernels.len]cuda.Function,
    stream: cuda.Stream,

    fn launch(
        runner: *const Runner,
        comptime index: usize,
        problem: *const Problem,
        causal: bool,
    ) !void {
        const Kernel = attention.kernels[index];
        const shape = problem.shape;
        const head_dim: f32 = @floatFromInt(shape.head_dim);
        const scale_log2: f32 = std.math.log2e / @sqrt(head_dim);
        try runner.functions[index].launch(.{
            .grid = .{
                .x = std.math.divCeil(u32, shape.seq, Kernel.block_m) catch unreachable,
                .y = shape.batch * shape.heads,
            },
            .block = .{ .x = Kernel.threads },
            .shared_memory_bytes = Kernel.dynamic_shared_bytes,
            .stream = runner.stream,
        }, .{
            problem.q,
            problem.k,
            problem.v,
            problem.o,
            shape.seq,
            scale_log2,
            @as(u32, @intFromBool(causal)),
        });
    }

    /// Fills each element of `timings` with the mean time in milliseconds of one launch, over
    /// `options.iterations` launches.
    fn measure(
        runner: *const Runner,
        comptime index: usize,
        problem: *const Problem,
        causal: bool,
        options: Options,
        timings: []f64,
    ) !void {
        for (0..options.warmup) |_| try runner.launch(index, problem, causal);
        try runner.stream.synchronize();
        const start = try cuda.Event.create();
        defer start.destroy();
        const stop = try cuda.Event.create();
        defer stop.destroy();
        const iterations: f64 = @floatFromInt(options.iterations);
        for (timings) |*milliseconds| {
            try start.record(runner.stream);
            for (0..options.iterations) |_| try runner.launch(index, problem, causal);
            try stop.record(runner.stream);
            try stop.synchronize();
            milliseconds.* = @as(f64, try start.elapsed(stop)) / iterations;
            if (!std.math.isFinite(milliseconds.*) or milliseconds.* <= 0) {
                return error.InvalidDeviceTiming;
            }
        }
    }

    /// Fills the output with NaN, launches the kernel once and compares the checked rows with the
    /// reference. A value that the kernel does not write stays NaN and fails the comparison.
    fn validate(
        runner: *const Runner,
        comptime index: usize,
        problem: *Problem,
        causal: bool,
    ) !Accuracy {
        try problem.o.memsetAsync(0xff, runner.stream);
        try runner.launch(index, problem, causal);
        try runner.stream.synchronize();
        return problem.check(causal);
    }
};

fn maskName(causal: bool) []const u8 {
    return if (causal) "causal" else "full";
}

/// Returns `name` without the prefix `flash_attn_`, so that the kernel column stays narrow.
fn shortName(name: []const u8) []const u8 {
    const prefix = "flash_attn_";
    return if (std.mem.startsWith(u8, name, prefix)) name[prefix.len..] else name;
}

const kernel_column = std.fmt.comptimePrint("{{s:<{d}}}", .{width: {
    var width = "Kernel".len;
    for (attention.kernels) |Kernel| width = @max(width, shortName(Kernel.name).len);
    break :width width;
}});
const kernel_cell = cell.cyan ++ kernel_column ++ cell.normal;
const mask_cell = cell.italic ++ "{s:<6}" ++ cell.normal;

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
    try renderer.row(
        .title,
        "{s}: O = softmax(Q K^T / sqrt(head_dim)) V, BF16 Q, K, V and O, FP32 accumulation\n",
        .{try device.name(&name)},
    );
    const module = try cuda.Module.load(@embedFile("attention_kernel.ptx"));
    defer module.unload();
    const stream = try cuda.Stream.create();
    defer stream.destroy();
    var runner: Runner = .{ .functions = undefined, .stream = stream };
    const kernel_format = kernel_cell ++ "  {d:>9}  {d:>11}  {d:>12}  {d:>14}";
    try renderer.row(.heading, kernel_column ++ "  {s:>9}  {s:>11}  {s:>12}  {s:>14}", .{
        "Kernel",
        "Registers",
        "Local bytes",
        "Shared bytes",
        "Work groups/SM",
    });
    inline for (attention.kernels, &runner.functions) |Kernel, *function| {
        function.* = try module.function(Kernel.name);
        try function.setMaxDynamicSharedMemory(Kernel.dynamic_shared_bytes);
        const static_shared: u32 = @intCast(try function.attribute(.shared_size_bytes));
        const shared = Kernel.dynamic_shared_bytes;
        try renderer.row(.body, kernel_format, .{
            shortName(Kernel.name),
            @as(u32, @intCast(try function.attribute(.num_regs))),
            @as(u32, @intCast(try function.attribute(.local_size_bytes))),
            static_shared + shared,
            try function.maxActiveBlocksPerMultiprocessor(Kernel.threads, shared),
        });
    }
    if (options.check != 0) {
        const check_format = "{d:>5}  {d:>5}  {d:>5}  {d:>4}  " ++ mask_cell ++ "  " ++
            kernel_cell ++ "  {d:>6}  {e:>9.3}  {e:>9.3}";
        try renderer.row(.title, "\nChecks of every output row against an f32 reference", .{});
        const check_heading = "{s:>5}  {s:>5}  {s:>5}  {s:>4}  {s:<6}  " ++ kernel_column ++
            "  {s:>6}  {s:>9}  {s:>9}";
        try renderer.row(.heading, check_heading, .{
            "Batch", "Heads", "Seq", "Dim", "Mask", "Kernel", "Rows", "Max error", "RMS error",
        });
        const small = [_]struct { batch: u32, heads: u32, seq: u32, drift: f32 }{
            .{ .batch = 1, .heads = 2, .seq = 128, .drift = 0 },
            .{ .batch = 2, .heads = 2, .seq = 512, .drift = 0 },
            .{ .batch = 1, .heads = 2, .seq = 200, .drift = 0 },
            .{ .batch = 2, .heads = 2, .seq = 77, .drift = 0 },
            .{ .batch = 1, .heads = 2, .seq = 448, .drift = 40 },
            .{ .batch = 1, .heads = 2, .seq = 333, .drift = -40 },
        };
        for (small) |dims| {
            for ([_]u32{ 64, 128 }) |head_dim| {
                const shape: Shape = .{
                    .batch = dims.batch,
                    .heads = dims.heads,
                    .seq = dims.seq,
                    .head_dim = head_dim,
                };
                var problem = try Problem.init(allocator, shape, options.seed, dims.drift, .all);
                defer problem.deinit(allocator);
                for ([_]bool{ false, true }) |causal| {
                    inline for (attention.kernels, 0..) |Kernel, index| {
                        if (Kernel.head_dim == head_dim) {
                            const accuracy = try runner.validate(index, &problem, causal);
                            try renderer.row(.body, check_format, .{
                                shape.batch,
                                shape.heads,
                                shape.seq,
                                shape.head_dim,
                                maskName(causal),
                                shortName(Kernel.name),
                                problem.rows.len,
                                accuracy.max_abs,
                                accuracy.rms,
                            });
                        }
                    }
                }
            }
        }
    }
    if (options.benchmark == 0) return;
    try renderer.row(.title, "\nBatch {d}, {d} heads, seed {d}: {d} warmup launches, " ++
        "{d} rounds of {d} samples of {d} launches", .{
        options.batch,
        options.heads,
        options.seed,
        options.warmup,
        options.rounds,
        options.repeat,
        options.iterations,
    });
    const benchmark_heading = "{s:>5}  {s:>4}  {s:<6}  " ++ kernel_column ++
        "  {s:>9}  {s:>6}  {s:>9}  {s:>9}  {s}";
    try renderer.row(.heading, benchmark_heading, .{
        "Seq", "Dim", "Mask", "Kernel", "Median ms", "TF/s", "Best TF/s", "Max error", "Round TF/s",
    });
    const all_seqs = [_]u32{ 1024, 4096 };
    const seqs: []const u32 = if (options.seq == 0) &all_seqs else &.{options.seq};
    for (seqs) |seq| {
        for ([_]u32{ 64, 128 }) |head_dim| {
            const shape: Shape = .{
                .batch = options.batch,
                .heads = options.heads,
                .seq = seq,
                .head_dim = head_dim,
            };
            var problem = try Problem.init(allocator, shape, options.seed, 0, .sampled);
            defer problem.deinit(allocator);
            for ([_]bool{ false, true }) |causal| {
                try benchmark(allocator, &runner, &problem, causal, options, &renderer);
            }
        }
    }
}

/// Validates and times every kernel for the head dimension of `problem`. The kernels take turns
/// in each round, so that a slow drift of the GPU speed affects all of them alike. The function
/// checks the output after the timed launches of each kernel. On a terminal, the rows redraw after
/// each round, and the row of the kernel with the lowest median time is highlighted.
fn benchmark(
    allocator: std.mem.Allocator,
    runner: *const Runner,
    problem: *Problem,
    causal: bool,
    options: Options,
    renderer: *TerminalTableRenderer,
) !void {
    const samples = options.rounds * options.repeat;
    const timings = try allocator.alloc(f64, (attention.kernels.len + 1) * samples);
    defer allocator.free(timings);
    const scratch = timings[attention.kernels.len * samples ..];
    var max_errors: [attention.kernels.len]f64 = undefined;
    inline for (attention.kernels, 0..) |Kernel, index| {
        if (Kernel.head_dim == problem.shape.head_dim) {
            max_errors[index] = (try runner.validate(index, problem, causal)).max_abs;
        }
    }
    renderer.keep();
    for (0..options.rounds + 1) |rounds_done| {
        if (rounds_done != 0) {
            const round = rounds_done - 1;
            inline for (attention.kernels, 0..) |Kernel, index| {
                if (Kernel.head_dim == problem.shape.head_dim) {
                    const first = index * samples + round * options.repeat;
                    const round_timings = timings[first..][0..options.repeat];
                    try runner.measure(index, problem, causal, options, round_timings);
                    _ = try problem.check(causal);
                }
            }
        }
        if (renderer.terminal or rounds_done == options.rounds) {
            const measured_samples = rounds_done * options.repeat;
            var fastest: ?usize = null;
            var fastest_time = std.math.inf(f64);
            inline for (attention.kernels, 0..) |Kernel, index| {
                if (Kernel.head_dim == problem.shape.head_dim and measured_samples != 0) {
                    const time = median(timings[index * samples ..][0..measured_samples], scratch);
                    if (time < fastest_time) {
                        fastest = index;
                        fastest_time = time;
                    }
                }
            }
            try renderer.rewind();
            inline for (attention.kernels, 0..) |Kernel, index| {
                if (Kernel.head_dim == problem.shape.head_dim) {
                    const measured = timings[index * samples ..][0..measured_samples];
                    const look: TerminalTableRenderer.Look =
                        if (fastest == index) .highlight else .body;
                    try report(renderer, look, Kernel.name, problem.shape, causal, .{
                        .timings = measured,
                        .repeat = options.repeat,
                        .max_error = max_errors[index],
                        .scratch = scratch,
                    });
                }
            }
        }
    }
}

const benchmark_key = "{d:>5}  {d:>4}  " ++ mask_cell ++ "  " ++ kernel_cell;
const benchmark_format = benchmark_key ++ "  {d:>9.4}  " ++ cell.bold ++ "{d:>6.2}" ++
    cell.normal ++ "  {d:>9.2}  " ++ cell.dim ++ "{e:>9.3}{s}" ++ cell.normal;

const Measurements = struct {
    timings: []const f64,
    repeat: u32,
    max_error: f64,
    scratch: []f64,
};

/// Writes the table row of one kernel. A kernel without timings shows `measuring` in place of the
/// numbers.
fn report(
    renderer: *TerminalTableRenderer,
    look: TerminalTableRenderer.Look,
    name: []const u8,
    shape: Shape,
    causal: bool,
    measurements: Measurements,
) !void {
    const timings = measurements.timings;
    const columns = .{ shape.seq, shape.head_dim, maskName(causal), shortName(name) };
    if (timings.len == 0) {
        const measuring = cell.dim ++ cell.italic ++ "measuring" ++ cell.normal;
        return renderer.row(look, benchmark_key ++ "  " ++ measuring, columns);
    }
    const operations = shape.operations(causal);
    var text: [16 * 8]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&text);
    var first: usize = 0;
    while (first < timings.len) : (first += measurements.repeat) {
        const round_median = median(timings[first..][0..measurements.repeat], measurements.scratch);
        try writer.print(" {d:>6.2}", .{operations / (round_median * 1e9)});
    }
    const overall = median(timings, measurements.scratch);
    try renderer.row(look, benchmark_format, columns ++ .{
        overall,
        operations / (overall * 1e9),
        operations / (std.mem.min(f64, timings) * 1e9),
        measurements.max_error,
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
    if (options.warmup > 10000 or
        options.iterations == 0 or options.iterations > 10000 or
        options.repeat == 0 or options.repeat > 1000 or
        options.rounds == 0 or options.rounds > 16)
        return error.InvalidBenchmarkCount;
    if (options.batch == 0 or options.heads == 0 or options.seq > 65536 or
        @as(u64, options.batch) * options.heads > 65535)
        return error.InvalidBenchmarkSize;
    return options;
}

const Row = struct { head: u32, index: u32 };

/// `all` selects every row for `Problem.check`, and `sampled` selects 32 rows of 3 heads.
const RowSelection = enum { all, sampled };

/// A problem holds random bf16 inputs, an output buffer, and the f32 reference rows of full and
/// causal attention.
const Problem = struct {
    shape: Shape,
    q: cuda.Buffer(u16),
    k: cuda.Buffer(u16),
    v: cuda.Buffer(u16),
    o: cuda.Buffer(u16),
    computed: []u16,
    rows: []Row,
    /// Index 0 holds the reference rows of full attention, and index 1 those of causal attention.
    expected: [2][]f32,
    /// For each element of `expected`, a magnitude is the sum of p * |v| over the keys divided by
    /// the sum of p. The error from rounding p to bf16 in the sum of p * v scales with the
    /// magnitude.
    magnitudes: [2][]f32,

    /// A nonzero `drift` shifts the scores by about `drift` from the first key to the last key,
    /// so that the maximum of a row changes along the keys.
    fn init(
        allocator: std.mem.Allocator,
        shape: Shape,
        seed: u64,
        drift: f32,
        selection: RowSelection,
    ) !Problem {
        const count = shape.elements();
        const host = try allocator.alloc(u16, 3 * count);
        defer allocator.free(host);
        const q_host = host[0..count];
        const k_host = host[count..][0..count];
        const v_host = host[2 * count ..][0..count];
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        const head_dim: f32 = @floatFromInt(shape.head_dim);
        const seq: f32 = @floatFromInt(shape.seq);
        const q_bias: f32 = if (drift == 0) 0 else 1;
        for (q_host) |*element| element.* = toBf16(random.floatNorm(f32) + q_bias);
        for (k_host, 0..) |*element, index| {
            const key: f32 = @floatFromInt(index / shape.head_dim % shape.seq);
            const bias = drift * key / seq / @sqrt(head_dim);
            element.* = toBf16(random.floatNorm(f32) + bias);
        }
        for (v_host) |*element| element.* = toBf16(random.floatNorm(f32));
        const q = try cuda.dupe(u16, q_host);
        errdefer q.free();
        const k = try cuda.dupe(u16, k_host);
        errdefer k.free();
        const v = try cuda.dupe(u16, v_host);
        errdefer v.free();
        const o = try cuda.alloc(u16, count);
        errdefer o.free();
        const computed = try allocator.alloc(u16, count);
        errdefer allocator.free(computed);
        const rows = try selectRows(allocator, shape, selection);
        errdefer allocator.free(rows);
        const reference = try allocator.alloc(f32, 4 * rows.len * shape.head_dim);
        errdefer allocator.free(reference);
        var expected: [2][]f32 = undefined;
        var magnitudes: [2][]f32 = undefined;
        const row_elements = rows.len * shape.head_dim;
        const inputs: Inputs = .{ .q = q_host, .k = k_host, .v = v_host };
        for (&expected, &magnitudes, 0..) |*values, *magnitude, mask| {
            values.* = reference[2 * mask * row_elements ..][0..row_elements];
            magnitude.* = reference[(2 * mask + 1) * row_elements ..][0..row_elements];
            const causal = mask == 1;
            try computeReference(allocator, shape, inputs, rows, causal, values.*, magnitude.*);
        }
        return .{
            .shape = shape,
            .q = q,
            .k = k,
            .v = v,
            .o = o,
            .computed = computed,
            .rows = rows,
            .expected = expected,
            .magnitudes = magnitudes,
        };
    }

    fn deinit(problem: *Problem, allocator: std.mem.Allocator) void {
        problem.q.free();
        problem.k.free();
        problem.v.free();
        problem.o.free();
        allocator.free(problem.computed);
        allocator.free(problem.rows);
        allocator.free(problem.expected[0].ptr[0 .. 4 * problem.expected[0].len]);
    }

    /// Copies the output to the host, compares the checked rows with the reference and returns
    /// the maximum and RMS errors.
    fn check(problem: *Problem, causal: bool) !Accuracy {
        try problem.o.copyTo(problem.computed);
        const shape = problem.shape;
        const expected = problem.expected[@intFromBool(causal)];
        const magnitudes = problem.magnitudes[@intFromBool(causal)];
        var max_error: f64 = 0;
        var squared_error: f64 = 0;
        var failures: usize = 0;
        for (problem.rows, 0..) |row, row_index| {
            const first = (@as(usize, row.head) * shape.seq + row.index) * shape.head_dim;
            const actual_row = problem.computed[first..][0..shape.head_dim];
            const first_expected = row_index * shape.head_dim;
            const expected_row = expected[first_expected..][0..shape.head_dim];
            const magnitude_row = magnitudes[first_expected..][0..shape.head_dim];
            for (actual_row, expected_row, magnitude_row, 0..) |bits, reference, size, column| {
                const actual = fromBf16(bits);
                const difference = @abs(@as(f64, actual) - reference);
                max_error = @max(max_error, difference);
                squared_error += difference * difference;
                // 1e-2 covers the bf16 rounding of the output, and 4e-3 that of the probabilities.
                const tolerance = 1e-2 * @abs(@as(f64, reference)) + 4e-3 * @as(f64, size);
                if (!std.math.isFinite(actual) or difference > tolerance) {
                    if (failures < 4) std.log.err(
                        "Mismatch head {d} row {d} column {d}: actual={d}, reference={d}",
                        .{ row.head, row.index, column, actual, reference },
                    );
                    failures += 1;
                }
            }
        }
        const checked = problem.rows.len * shape.head_dim;
        if (failures != 0) {
            std.log.err("{d}/{d} outputs failed correctness", .{ failures, checked });
            return error.IncorrectAttention;
        }
        return .{
            .max_abs = max_error,
            .rms = @sqrt(squared_error / @as(f64, @floatFromInt(checked))),
        };
    }
};

const Accuracy = struct { max_abs: f64, rms: f64 };

fn selectRows(allocator: std.mem.Allocator, shape: Shape, selection: RowSelection) ![]Row {
    const head_count = shape.batch * shape.heads;
    switch (selection) {
        .all => {
            const rows = try allocator.alloc(Row, @as(usize, head_count) * shape.seq);
            for (rows, 0..) |*row, index| row.* = .{
                .head = @intCast(index / shape.seq),
                .index = @intCast(index % shape.seq),
            };
            return rows;
        },
        .sampled => {
            const heads = [_]u32{ 0, head_count / 2, head_count - 1 };
            const per_head = 32;
            const rows = try allocator.alloc(Row, heads.len * per_head);
            for (heads, 0..) |head, head_index| {
                for (0..per_head) |sample| {
                    const spread: u32 = @intCast(sample * shape.seq / per_head + sample % 7);
                    const index = if (sample + 1 == per_head) shape.seq - 1 else spread;
                    rows[head_index * per_head + sample] = .{
                        .head = head,
                        .index = @min(index, shape.seq - 1),
                    };
                }
            }
            return rows;
        },
    }
}

const Inputs = struct { q: []const u16, k: []const u16, v: []const u16 };

/// Computes the f32 reference output and magnitudes of each selected row from the bf16 inputs.
fn computeReference(
    allocator: std.mem.Allocator,
    shape: Shape,
    inputs: Inputs,
    rows: []const Row,
    causal: bool,
    expected: []f32,
    magnitudes: []f32,
) !void {
    const head_elements = @as(usize, shape.seq) * shape.head_dim;
    const buffers = try allocator.alloc(f32, 2 * head_elements + shape.seq + shape.head_dim);
    defer allocator.free(buffers);
    const keys = buffers[0..head_elements];
    const values = buffers[head_elements..][0..head_elements];
    const scores = buffers[2 * head_elements ..][0..shape.seq];
    const query = buffers[2 * head_elements + shape.seq ..][0..shape.head_dim];
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(shape.head_dim)));
    var loaded_head: ?u32 = null;
    for (rows, 0..) |row, row_index| {
        const head_first = @as(usize, row.head) * head_elements;
        if (loaded_head != row.head) {
            const k = inputs.k[head_first..][0..head_elements];
            const v = inputs.v[head_first..][0..head_elements];
            for (keys, k) |*dst, src| dst.* = fromBf16(src);
            for (values, v) |*dst, src| dst.* = fromBf16(src);
            loaded_head = row.head;
        }
        const query_first = head_first + @as(usize, row.index) * shape.head_dim;
        const q = inputs.q[query_first..][0..shape.head_dim];
        for (query, q) |*dst, src| dst.* = fromBf16(src);
        const key_count = if (causal) row.index + 1 else shape.seq;
        var max: f32 = -std.math.inf(f32);
        for (scores[0..key_count], 0..) |*score, key| {
            score.* = dot(query, keys[key * shape.head_dim ..][0..shape.head_dim]) * scale;
            max = @max(max, score.*);
        }
        const output = expected[row_index * shape.head_dim ..][0..shape.head_dim];
        const magnitude = magnitudes[row_index * shape.head_dim ..][0..shape.head_dim];
        @memset(output, 0);
        @memset(magnitude, 0);
        var sum: f32 = 0;
        for (scores[0..key_count], 0..) |score, key| {
            const p = @exp(score - max);
            sum += p;
            const value = values[key * shape.head_dim ..][0..shape.head_dim];
            for (output, magnitude, value) |*out, *size, element| {
                out.* += p * element;
                size.* += p * @abs(element);
            }
        }
        for (output, magnitude) |*out, *size| {
            out.* /= sum;
            size.* /= sum;
        }
    }
}

fn dot(a: []const f32, b: []const f32) f32 {
    const Lanes = @Vector(8, f32);
    var sum: Lanes = @splat(0);
    var index: usize = 0;
    while (index < a.len) : (index += 8) {
        const x: Lanes = a[index..][0..8].*;
        const y: Lanes = b[index..][0..8].*;
        sum += x * y;
    }
    return @reduce(.Add, sum);
}

fn toBf16(number: f32) u16 {
    const bits: u32 = @bitCast(number);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

fn fromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
