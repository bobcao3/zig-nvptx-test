const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;
const Cuda = @import("cuda");

const usage =
    \\Usage: zig-nvptx-behavior --zig-src=<dir> --zig=<exe> --executor=<exe> --test-runner=<file>
    \\           --out=<file.md> [--mcpu=<cpu>]... [--jobs=<n>] [--ptxas=<exe>] [file.zig...]
    \\
    \\Runs each file that test/behavior.zig of the Zig checkout <dir> imports as its own `zig test`
    \\module for each --mcpu target on the first CUDA device and writes a Markdown table of the
    \\results to <file.md>. Without --mcpu, the targets are sm_80 and the compute capability of the
    \\device. A file whose target the device cannot load is checked with ptxas and not run.
    \\Each module uses the test runner <file> and runs through the executor <exe>. With file
    \\arguments, only the behavior files with these names run.
    \\
;

const Row = struct {
    path: []const u8,
    status: []const u8 = "",
    passed: u32 = 0,
    skipped: u32 = 0,
    failed: u32 = 0,
    first_error: []const u8 = "",
};

const Job = struct {
    io: Io,
    repo: []const u8,
    zig: []const u8,
    executor: []const u8,
    test_runner: []const u8,
    mcpu_arg: []const u8 = "",
    ptxas_arg: ?[]const u8,
    rows: []Row,
    next: std.atomic.Value(usize) = .init(0),
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    var args = try init.minimal.args.iterateAllocator(arena);
    _ = args.skip();
    var repo: ?[]const u8 = null;
    var zig: ?[]const u8 = null;
    var executor: ?[]const u8 = null;
    var test_runner: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var targets: std.ArrayList([]const u8) = .empty;
    var jobs: usize = 4;
    var ptxas: ?[]const u8 = null;
    var filters: std.ArrayList([]const u8) = .empty;
    while (args.next()) |arg| {
        if (std.mem.cutPrefix(u8, arg, "--zig-src=")) |value| {
            repo = value;
        } else if (std.mem.cutPrefix(u8, arg, "--zig=")) |value| {
            zig = value;
        } else if (std.mem.cutPrefix(u8, arg, "--executor=")) |value| {
            executor = value;
        } else if (std.mem.cutPrefix(u8, arg, "--test-runner=")) |value| {
            test_runner = value;
        } else if (std.mem.cutPrefix(u8, arg, "--out=")) |value| {
            out_path = value;
        } else if (std.mem.cutPrefix(u8, arg, "--mcpu=")) |value| {
            try targets.append(arena, value);
        } else if (std.mem.cutPrefix(u8, arg, "--jobs=")) |value| {
            jobs = try std.fmt.parseUnsigned(usize, value, 10);
        } else if (std.mem.cutPrefix(u8, arg, "--ptxas=")) |value| {
            ptxas = value;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            fatal("unrecognized argument: {s}\n{s}", .{ arg, usage });
        } else {
            try filters.append(arena, std.fs.path.basename(arg));
        }
    }
    // Each `zig test` runs in the Zig checkout, so the paths that name a file must not be relative.
    const cwd = try std.process.currentPathAlloc(io, arena);
    for ([_]*?[]const u8{ &zig, &executor, &test_runner, &ptxas }) |path| {
        const name = path.* orelse continue;
        if (std.fs.path.dirname(name) != null) path.* = try std.fs.path.resolveAlloc(arena, &.{ cwd, name });
    }
    if (zig == null or executor == null or test_runner == null or out_path == null) fatal(usage, .{});

    var repo_dir = try Io.Dir.cwd().openDir(io, repo orelse fatal(usage, .{}), .{});
    defer repo_dir.close(io);
    const behavior = try repo_dir.readFileAlloc(io, "test/behavior.zig", arena, .unlimited);

    var paths: std.ArrayList([]const u8) = .empty;
    var rest: []const u8 = behavior;
    while (std.mem.find(u8, rest, "@import(\"behavior/")) |start| {
        rest = rest[start + "@import(\"".len ..];
        const end = std.mem.findScalar(u8, rest, '"') orelse break;
        const path = rest[0..end];
        if (std.mem.endsWith(u8, path, ".zig") and selected(filters.items, path)) try paths.append(arena, path);
        rest = rest[end..];
    }
    std.mem.sort([]const u8, paths.items, {}, lessThan);
    const unique_len = uniqueLen(paths.items);

    if (targets.items.len == 0) try targets.appendSlice(arena, try defaultTargets(arena));

    var job: Job = .{
        .io = io,
        .repo = repo.?,
        .zig = zig.?,
        .executor = executor.?,
        .test_runner = test_runner.?,
        .ptxas_arg = if (ptxas) |path| try std.fmt.allocPrint(arena, "--ptxas={s}", .{path}) else null,
        .rows = undefined,
    };
    var out: Io.Writer.Allocating = .init(arena);
    const w = &out.writer;
    try w.print("# Behavior tests on the GPU (nvptx64-cuda, Debug)\n\n", .{});
    for (targets.items) |mcpu| {
        const rows = try arena.alloc(Row, unique_len);
        for (rows, paths.items[0..unique_len]) |*row, path| row.* = .{ .path = path };
        job.rows = rows;
        job.mcpu_arg = try std.fmt.allocPrint(arena, "-mcpu={s}", .{mcpu});
        job.next.store(0, .monotonic);
        std.debug.print("{s}\n", .{mcpu});
        var group: Io.Group = .init;
        for (0..jobs) |_| try group.concurrent(io, worker, .{&job});
        try group.await(io);
        try writeSection(arena, w, mcpu, rows);
    }
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path.?, .data = out.written() });
    std.debug.print("{s}", .{out.written()});
}

/// Returns sm_80 and the exact target of the first device. Without a CUDA driver, only sm_80.
fn defaultTargets(arena: std.mem.Allocator) ![]const []const u8 {
    const cuda = Cuda.Raw.load() catch return &.{"sm_80"};
    cuda.initDevice();
    const capability = cuda.computeCapability();
    const own = try std.fmt.allocPrint(arena, "sm_{d}{d}", .{ capability.major, capability.minor });
    if (std.mem.eql(u8, own, "sm_80")) return &.{"sm_80"};
    const both = try arena.alloc([]const u8, 2);
    both[0] = "sm_80";
    both[1] = own;
    return both;
}

fn writeSection(arena: std.mem.Allocator, w: *Io.Writer, mcpu: []const u8, rows: []const Row) !void {
    var totals: [3]u32 = @splat(0);
    var statuses: std.StringArrayHashMapUnmanaged(u32) = .empty;
    for (rows) |row| {
        totals[0] += row.passed;
        totals[1] += row.skipped;
        totals[2] += row.failed;
        const entry = try statuses.getOrPutValue(arena, row.status, 0);
        entry.value_ptr.* += 1;
    }
    try w.print("## {s}\n\nFiles: {d}", .{ mcpu, rows.len });
    for (statuses.keys(), statuses.values()) |status, count| try w.print("; {s}: {d}", .{ status, count });
    try w.print("\n\nTests in files that ran: {d} passed, {d} skipped, {d} failed\n\n", .{
        totals[0], totals[1], totals[2],
    });
    try w.print("| file | compiles? | passed | failed | skipped | first error |\n", .{});
    try w.print("|---|---|---|---|---|---|\n", .{});
    for (rows) |row| {
        try w.print("| {s} | {s} | {d} | {d} | {d} | ", .{
            row.path, row.status, row.passed, row.failed, row.skipped,
        });
        for (row.first_error[0..@min(row.first_error.len, 200)]) |byte| {
            if (byte == '|') try w.writeByte('\\');
            try w.writeByte(byte);
        }
        try w.print(" |\n", .{});
    }
    try w.writeByte('\n');
}

fn selected(filters: []const []const u8, path: []const u8) bool {
    if (filters.len == 0) return true;
    for (filters) |filter| {
        if (std.mem.eql(u8, filter, std.fs.path.basename(path))) return true;
    }
    return false;
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

fn uniqueLen(sorted: [][]const u8) usize {
    if (sorted.len == 0) return 0;
    var len: usize = 1;
    for (sorted[1..]) |path| {
        if (std.mem.eql(u8, path, sorted[len - 1])) continue;
        sorted[len] = path;
        len += 1;
    }
    return len;
}

fn worker(job: *Job) void {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    while (true) {
        const index = job.next.fetchAdd(1, .monotonic);
        if (index >= job.rows.len) return;
        const row = &job.rows[index];
        for (0..3) |_| {
            classify(job, arena_state.allocator(), row) catch |err| {
                row.status = "harness error";
                row.first_error = @errorName(err);
            };
            if (std.mem.find(u8, row.first_error, "CUDA_ERROR_OUT_OF_MEMORY") == null) break;
        }
        row.first_error = std.heap.page_allocator.dupe(u8, row.first_error) catch "";
        std.debug.print("{s}: {s}\n", .{ row.path, row.status });
    }
}

fn classify(job: *Job, arena: std.mem.Allocator, row: *Row) !void {
    const io = job.io;
    const test_path = try std.fmt.allocPrint(arena, "test/{s}", .{row.path});
    var timed_out = false;
    const output: []const u8 = output: {
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(arena, &.{
            job.zig,         "test",          test_path,
            "--test-runner", job.test_runner, "-target",
            "nvptx64-cuda",  job.mcpu_arg,    "--zig-lib-dir",
            "lib",           "--test-cmd",    job.executor,
        });
        try argv.append(arena, "-fno-llvm");
        if (job.ptxas_arg) |arg| try argv.appendSlice(arena, &.{ "--test-cmd", arg });
        try argv.append(arena, "--test-cmd-bin");
        const result = std.process.run(arena, io, .{
            .argv = argv.items,
            .cwd = .{ .path = job.repo },
            .timeout = .{ .duration = .{ .raw = .fromSeconds(600), .clock = .awake } },
        }) catch |err| switch (err) {
            error.Timeout => {
                timed_out = true;
                break :output "";
            },
            else => |e| return e,
        };
        break :output try std.mem.concat(arena, u8, &.{ result.stdout, result.stderr });
    };

    var ran = false;
    var finished = false;
    var panic_line: ?[]const u8 = null;
    var context_line: []const u8 = "";
    var location_line: ?[]const u8 = null;
    var ptxas_line: ?[]const u8 = null;
    var load_line: ?[]const u8 = null;
    var cuda_line: ?[]const u8 = null;
    var compile_error_line: ?[]const u8 = null;
    var any_error_line: ?[]const u8 = null;
    var fail_line: ?[]const u8 = null;
    var skip_line: ?[]const u8 = null;
    row.passed = 0;
    row.skipped = 0;
    row.failed = 0;

    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |line| {
        if (isTestLine(line)) ran = true;
        if (std.mem.cutPrefix(u8, line, "All ")) |rest| {
            if (std.mem.find(u8, rest, " tests passed")) |end| {
                row.passed = std.fmt.parseUnsigned(u32, rest[0..end], 10) catch 0;
                finished = true;
            }
        }
        if (parseSummary(line)) |counts| {
            row.passed, row.skipped, row.failed = counts;
            finished = true;
        }
        if (panic_line == null and std.mem.startsWith(u8, line, "thread ")) {
            if (std.mem.find(u8, line, " panic: ")) |at| panic_line = line[at + " panic: ".len ..];
        }
        if (context_line.len == 0 and std.mem.startsWith(u8, line, "Generating function")) context_line = line;
        if (location_line == null and (std.mem.find(u8, line, "src/codegen/nvptx") != null or
            std.mem.find(u8, line, "src/link/Nvptx") != null)) location_line = line;
        if (ptxas_line == null and std.mem.startsWith(u8, line, "ptxas ") and
            std.mem.find(u8, line, ", line ") != null) ptxas_line = try trimPtxPath(arena, line);
        if (load_line == null and (std.mem.find(u8, line, "cuModuleLoadDataEx failed") != null or
            std.mem.find(u8, line, "ptxas failed") != null)) load_line = line;
        if (cuda_line == null and std.mem.startsWith(u8, line, "error: cu") and
            std.mem.find(u8, line, " failed") != null) cuda_line = line;
        if (compile_error_line == null and std.mem.find(u8, line, ": error: ") != null) compile_error_line = line;
        if (any_error_line == null and std.mem.find(u8, line, "error") != null) any_error_line = line;
        if (fail_line == null and std.mem.find(u8, line, "FAIL") != null) fail_line = line;
        if (skip_line == null and std.mem.startsWith(u8, line, "skipped (")) skip_line = line;
    }

    if (!ran and (panic_line != null or std.mem.find(u8, output, "Compiler crash context") != null)) {
        row.status = "compiler crash";
        var text = try std.fmt.allocPrint(arena, "{s} ({s})", .{
            panic_line orelse "",
            std.mem.trim(u8, context_line, " "),
        });
        if (location_line) |line| {
            if (std.mem.find(u8, line, "src/")) |start| {
                const location = line[start..];
                const end = std.mem.findScalarPos(u8, location, (std.mem.findScalar(u8, location, ':') orelse 0) + 1, ':') orelse location.len;
                text = try std.fmt.allocPrint(arena, "{s} at {s}", .{ text, location[0..end] });
            }
        }
        row.first_error = text;
    } else if (load_line != null) {
        row.status = "ptx rejected";
        row.first_error = ptxas_line orelse load_line.?;
    } else if (skip_line != null) {
        row.status = "skipped";
        row.first_error = skip_line.?;
    } else if (!ran and cuda_line != null) {
        row.status = "executor error";
        row.first_error = cuda_line.?;
    } else if (!ran) {
        row.status = if (timed_out) "timeout" else "no";
        const line = compile_error_line orelse any_error_line orelse "no output";
        row.first_error = trimPath(line);
    } else {
        row.status = "yes";
        const fail = fail_line orelse "";
        if (timed_out) {
            row.first_error = try std.fmt.allocPrint(arena, "timeout; {s}", .{fail});
        } else if (!finished) {
            row.first_error = try std.fmt.allocPrint(arena, "executor did not finish; {s}", .{fail});
        } else {
            row.first_error = fail;
        }
    }
    row.first_error = std.mem.trim(u8, row.first_error, &std.ascii.whitespace);
}

fn isTestLine(line: []const u8) bool {
    const slash = std.mem.findScalar(u8, line, '/') orelse return false;
    const space = std.mem.findScalarPos(u8, line, slash, ' ') orelse return false;
    return allDigits(line[0..slash]) and allDigits(line[slash + 1 .. space]);
}

fn parseSummary(line: []const u8) ?[3]u32 {
    var it = std.mem.tokenizeAny(u8, line, " ;.");
    var counts: [3]u32 = undefined;
    inline for (.{ "passed", "skipped", "failed" }, 0..) |word, i| {
        counts[i] = std.fmt.parseUnsigned(u32, it.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, it.next() orelse return null, word)) return null;
    }
    return counts;
}

fn trimPath(line: []const u8) []const u8 {
    for ([_][]const u8{ "/test/", "/lib/" }) |marker| {
        if (std.mem.find(u8, line, marker)) |at| {
            if (std.mem.findScalar(u8, line[0..at], ' ') == null) return line[at + 1 ..];
        }
    }
    return line;
}

/// Replaces the path of the PTX file, which names a cache directory, so that the table does not
/// change from run to run.
fn trimPtxPath(arena: std.mem.Allocator, line: []const u8) ![]const u8 {
    const comma = std.mem.find(u8, line, ", line ") orelse return line;
    return std.fmt.allocPrint(arena, "ptxas test.ptx{s}", .{line[comma..]});
}

fn allDigits(text: []const u8) bool {
    if (text.len == 0) return false;
    for (text) |byte| if (!std.ascii.isDigit(byte)) return false;
    return true;
}
