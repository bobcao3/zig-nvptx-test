//! Compares the kernels of the Zig ports with the kernels that nvcc compiles from the upstream
//! samples, after one ptxas assembles both.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const fatal = std.process.fatal;

const usage =
    \\Usage: nvcc-compare (--out=<file.md> | --check) --arch=<sm> --zig-mode=<mode>
    \\           --nvcc-flags=<flags> [--skip <sample> <reason>]...
    \\           [--sample <sample> [--zig <info> <sass>]... [--nvcc <info> <sass>]...]...
    \\
    \\Reads the `ptxas -v` output <info> and the `nvdisasm -c` output <sass> of each PTX module
    \\of a sample. The modules of the Zig port follow --zig, and the modules that nvcc compiles
    \\from the upstream sample follow --nvcc. The tool pairs each Zig kernel with an nvcc kernel
    \\and compares their registers, local memory, shared memory and SASS instructions.
    \\
    \\--out writes the comparison to <file.md>. --check prints each Zig kernel that has no nvcc
    \\kernel, unless the upstream sample does not instantiate it, and each Zig kernel that uses
    \\local memory (a stack frame or spill stores) where its nvcc kernel uses none. It exits with
    \\status 1 when it prints a line.
    \\
;

const Stats = struct {
    registers: u32 = 0,
    stack: u32 = 0,
    spill_stores: u32 = 0,
    spill_loads: u32 = 0,
    shared: u32 = 0,
    instructions: u32 = 0,

    fn usesLocalMemory(stats: Stats) bool {
        return stats.stack != 0 or stats.spill_stores != 0;
    }
};

const metrics = [_]struct { field: []const u8, title: []const u8 }{
    .{ .field = "registers", .title = "Registers" },
    .{ .field = "stack", .title = "Stack frame (bytes)" },
    .{ .field = "spill_stores", .title = "Spill stores (bytes)" },
    .{ .field = "spill_loads", .title = "Spill loads (bytes)" },
    .{ .field = "shared", .title = "Shared memory (bytes)" },
    .{ .field = "instructions", .title = "SASS instructions" },
};

const Function = struct {
    symbol: []const u8,
    kernel: bool = false,
    stats: Stats = .{},
};

const Sample = struct {
    name: []const u8,
    zig: std.ArrayList(Function) = .empty,
    nvcc: std.ArrayList(Function) = .empty,
};

const Skip = struct {
    sample: []const u8,
    reason: []const u8,
};

/// A token of an nvcc kernel name that a Zig port spells differently. An alias with an empty
/// `sample` applies to every sample.
const Alias = struct {
    sample: []const u8 = "",
    nvcc: []const u8,
    zig: []const u8,
};

/// Zig kernels that pair with no nvcc kernel because the upstream sample does not instantiate
/// them. For 1024 threads, the upstream dispatch launches `reduce7<T, 1024, true>` whether or not
/// the size is a power of two.
const zig_only = [_]struct { sample: []const u8, kernel: []const u8 }{
    .{ .sample = "2_Concepts_and_Techniques/reduction", .kernel = "reduce7_i32_1024_any" },
    .{ .sample = "2_Concepts_and_Techniques/reduction", .kernel = "reduce7_f32_1024_any" },
    .{ .sample = "2_Concepts_and_Techniques/reduction", .kernel = "reduce7_f64_1024_any" },
};

fn isZigOnly(sample: []const u8, kernel: []const u8) bool {
    for (zig_only) |entry| {
        if (std.mem.eql(u8, entry.sample, sample) and std.mem.eql(u8, entry.kernel, kernel)) return true;
    }
    return false;
}

const aliases = [_]Alias{
    .{ .nvcc = "signed char", .zig = "i8" },
    .{ .nvcc = "unsigned char", .zig = "u8" },
    .{ .nvcc = "short", .zig = "i16" },
    .{ .nvcc = "unsigned short", .zig = "u16" },
    .{ .nvcc = "int", .zig = "i32" },
    .{ .nvcc = "unsigned int", .zig = "u32" },
    .{ .nvcc = "long long", .zig = "i64" },
    .{ .nvcc = "unsigned long long", .zig = "u64" },
    .{ .nvcc = "float", .zig = "f32" },
    .{ .nvcc = "double", .zig = "f64" },
    .{ .sample = "2_Concepts_and_Techniques/reduction", .nvcc = "cg_reduce", .zig = "reduce8" },
    .{ .sample = "2_Concepts_and_Techniques/reduction", .nvcc = "multi_warp_cg_reduce", .zig = "reduce9" },
    .{ .sample = "2_Concepts_and_Techniques/reduction", .nvcc = "true", .zig = "pow2" },
    .{ .sample = "2_Concepts_and_Techniques/reduction", .nvcc = "false", .zig = "any" },
    .{ .sample = "2_Concepts_and_Techniques/threadFenceReduction", .nvcc = "true", .zig = "pow2" },
    .{ .sample = "2_Concepts_and_Techniques/threadFenceReduction", .nvcc = "false", .zig = "any" },
    .{ .sample = "6_Performance/alignedTypes", .nvcc = "unsigned char", .zig = "uint8" },
    .{ .sample = "6_Performance/alignedTypes", .nvcc = "unsigned short", .zig = "uint16" },
    .{ .sample = "6_Performance/alignedTypes", .nvcc = "unsigned int", .zig = "i32" },
};

/// A kernel name of nvcc: the name of the function and its template arguments.
const Name = struct {
    text: []const u8,
    base: []const u8,
    arguments: []const []const u8 = &.{},
};

const Pair = struct {
    zig: *const Function,
    nvcc: *const Function,
    nvcc_name: Name,
};

const Comparison = struct {
    sample: *const Sample,
    pairs: []const Pair,
    zig_unpaired: []const *const Function,
    nvcc_unpaired: []const Name,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var out_path: ?[]const u8 = null;
    var check = false;
    var arch: []const u8 = "";
    var zig_mode: []const u8 = "";
    var nvcc_flags: []const u8 = "";
    var skips: std.ArrayList(Skip) = .empty;
    var samples: std.ArrayList(Sample) = .empty;
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.cutPrefix(u8, arg, "--out=")) |value| {
            out_path = value;
        } else if (std.mem.eql(u8, arg, "--check")) {
            check = true;
        } else if (std.mem.cutPrefix(u8, arg, "--arch=")) |value| {
            arch = value;
        } else if (std.mem.cutPrefix(u8, arg, "--zig-mode=")) |value| {
            zig_mode = value;
        } else if (std.mem.cutPrefix(u8, arg, "--nvcc-flags=")) |value| {
            nvcc_flags = value;
        } else if (std.mem.eql(u8, arg, "--skip") and index + 2 < args.len) {
            try skips.append(arena, .{ .sample = args[index + 1], .reason = args[index + 2] });
            index += 2;
        } else if (std.mem.eql(u8, arg, "--sample") and index + 1 < args.len) {
            try samples.append(arena, .{ .name = args[index + 1] });
            index += 1;
        } else if ((std.mem.eql(u8, arg, "--zig") or std.mem.eql(u8, arg, "--nvcc")) and
            index + 2 < args.len and samples.items.len != 0)
        {
            const sample = &samples.items[samples.items.len - 1];
            const functions = if (std.mem.eql(u8, arg, "--zig")) &sample.zig else &sample.nvcc;
            const info = try Io.Dir.cwd().readFileAlloc(io, args[index + 1], arena, .unlimited);
            const sass = try Io.Dir.cwd().readFileAlloc(io, args[index + 2], arena, .unlimited);
            try readModule(arena, functions, info, sass);
            index += 2;
        } else {
            fatal("unrecognized argument: {s}\n{s}", .{ arg, usage });
        }
    }
    if (out_path == null and !check) fatal("{s}", .{usage});

    const comparisons = try arena.alloc(Comparison, samples.items.len);
    for (comparisons, samples.items) |*comparison, *sample| comparison.* = try pairKernels(arena, sample);

    if (out_path) |path| {
        var out: Io.Writer.Allocating = .init(arena);
        try writeReport(&out.writer, .{
            .arch = arch,
            .zig_mode = zig_mode,
            .nvcc_flags = nvcc_flags,
        }, comparisons, skips.items);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.written() });
    }
    if (check) {
        var out: Io.Writer.Allocating = .init(arena);
        const w = &out.writer;
        for (comparisons) |comparison| {
            const sample = comparison.sample.name;
            for (comparison.zig_unpaired) |function| {
                if (isZigOnly(sample, function.symbol)) continue;
                try w.print("{s}: no nvcc kernel pairs with the Zig kernel {s}\n", .{ sample, function.symbol });
            }
            for (comparison.pairs) |pair| {
                const zig = pair.zig.stats;
                if (!zig.usesLocalMemory() or pair.nvcc.stats.usesLocalMemory()) continue;
                try w.print("{s}: {s} has a stack frame of {d} bytes and {d} bytes of spill stores, " ++
                    "and the nvcc kernel {s} has neither\n", .{
                    sample, pair.zig.symbol, zig.stack, zig.spill_stores, pair.nvcc_name.text,
                });
            }
        }
        if (out.written().len != 0) {
            try Io.File.stderr().writeStreamingAll(io, out.written());
            std.process.exit(1);
        }
    }
}

/// Adds the functions of one PTX module to `functions`.
fn readModule(arena: Allocator, functions: *std.ArrayList(Function), info: []const u8, sass: []const u8) !void {
    const first = functions.items.len;
    var compiling: ?usize = null;
    var properties: ?usize = null;
    var lines = std.mem.tokenizeScalar(u8, info, '\n');
    while (lines.next()) |line| {
        if (std.mem.cut(u8, line, "Compiling entry function '")) |parts| {
            const index = try functionIndex(arena, functions, first, std.mem.sliceTo(parts[1], '\''));
            functions.items[index].kernel = true;
            compiling = index;
        } else if (std.mem.cut(u8, line, "Function properties for ")) |parts| {
            properties = try functionIndex(arena, functions, first, std.mem.trim(u8, parts[1], " \r"));
        } else if (std.mem.indexOf(u8, line, " bytes stack frame") != null) {
            const stats = &functions.items[properties orelse continue].stats;
            stats.stack = numberBefore(line, " bytes stack frame");
            stats.spill_stores = numberBefore(line, " bytes spill stores");
            stats.spill_loads = numberBefore(line, " bytes spill loads");
        } else if (std.mem.indexOf(u8, line, ": Used ") != null) {
            const stats = &functions.items[compiling orelse continue].stats;
            stats.registers = numberBefore(line, " registers");
            stats.shared = numberBefore(line, " bytes smem");
        }
    }

    var current: ?usize = null;
    lines = std.mem.tokenizeScalar(u8, sass, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.cutPrefix(u8, line, ".section\t.text.")) |rest| {
            const symbol = rest[0 .. std.mem.indexOfScalar(u8, rest, ',') orelse rest.len];
            current = try functionIndex(arena, functions, first, std.mem.trim(u8, symbol, "\""));
        } else if (std.mem.startsWith(u8, line, "/*")) {
            const parts = std.mem.cut(u8, line, "*/") orelse continue;
            var words = std.mem.tokenizeAny(u8, parts[1], " \t;");
            var mnemonic = words.next() orelse continue;
            if (mnemonic[0] == '@') mnemonic = words.next() orelse continue;
            if (std.mem.eql(u8, mnemonic, "NOP")) continue;
            functions.items[current orelse continue].stats.instructions += 1;
        }
    }
}

fn functionIndex(arena: Allocator, functions: *std.ArrayList(Function), first: usize, symbol: []const u8) !usize {
    for (functions.items[first..], first..) |function, index| {
        if (std.mem.eql(u8, function.symbol, symbol)) return index;
    }
    try functions.append(arena, .{ .symbol = symbol });
    return functions.items.len - 1;
}

fn numberBefore(line: []const u8, suffix: []const u8) u32 {
    const end = std.mem.indexOf(u8, line, suffix) orelse return 0;
    var start = end;
    while (start > 0 and std.ascii.isDigit(line[start - 1])) start -= 1;
    return std.fmt.parseUnsigned(u32, line[start..end], 10) catch 0;
}

/// Pairs each Zig kernel with the nvcc kernel whose name, with the leading template arguments,
/// equals the Zig name once both drop the case and every character that is not a letter or a
/// digit. A pairing with more template arguments wins. A Zig kernel that two nvcc kernels match
/// equally stays unpaired.
fn pairKernels(arena: Allocator, sample: *const Sample) !Comparison {
    const Candidate = struct {
        function: *const Function,
        name: Name,
        keys: []const []const u8,
        paired: bool = false,
    };
    var candidates: std.ArrayList(Candidate) = .empty;
    for (sample.nvcc.items) |*function| {
        if (!function.kernel) continue;
        const name = try demangle(arena, function.symbol);
        const keys = try arena.alloc([]const u8, name.arguments.len + 1);
        for (keys, 0..) |*key, count| key.* = try nameKey(arena, sample.name, name, count);
        try candidates.append(arena, .{ .function = function, .name = name, .keys = keys });
    }

    var pairs: std.ArrayList(Pair) = .empty;
    var zig_unpaired: std.ArrayList(*const Function) = .empty;
    for (sample.zig.items) |*function| {
        if (!function.kernel) continue;
        const key = try normalize(arena, function.symbol);
        var best: ?usize = null;
        var best_count: usize = 0;
        var ambiguous = false;
        for (candidates.items, 0..) |candidate, index| {
            if (candidate.paired) continue;
            var count = candidate.keys.len;
            const matched = while (count > 0) {
                count -= 1;
                if (std.mem.eql(u8, candidate.keys[count], key)) break true;
            } else false;
            if (!matched) continue;
            if (best == null or count > best_count) {
                best = index;
                best_count = count;
                ambiguous = false;
            } else if (count == best_count) {
                ambiguous = true;
            }
        }
        if (best == null or ambiguous) {
            try zig_unpaired.append(arena, function);
            continue;
        }
        const candidate = &candidates.items[best.?];
        candidate.paired = true;
        try pairs.append(arena, .{ .zig = function, .nvcc = candidate.function, .nvcc_name = candidate.name });
    }

    var nvcc_unpaired: std.ArrayList(Name) = .empty;
    for (candidates.items) |candidate| {
        if (!candidate.paired) try nvcc_unpaired.append(arena, candidate.name);
    }
    return .{
        .sample = sample,
        .pairs = pairs.items,
        .zig_unpaired = zig_unpaired.items,
        .nvcc_unpaired = nvcc_unpaired.items,
    };
}

fn nameKey(arena: Allocator, sample: []const u8, name: Name, argument_count: usize) ![]const u8 {
    var key: std.ArrayList(u8) = .empty;
    try appendNormalized(arena, &key, alias(sample, name.base));
    for (name.arguments[0..argument_count]) |argument| try appendNormalized(arena, &key, alias(sample, argument));
    return key.items;
}

fn alias(sample: []const u8, token: []const u8) []const u8 {
    for ([_][]const u8{ sample, "" }) |scope| {
        for (aliases) |entry| {
            if (std.mem.eql(u8, entry.sample, scope) and std.mem.eql(u8, entry.nvcc, token)) return entry.zig;
        }
    }
    return token;
}

fn normalize(arena: Allocator, text: []const u8) ![]const u8 {
    var key: std.ArrayList(u8) = .empty;
    try appendNormalized(arena, &key, text);
    return key.items;
}

fn appendNormalized(arena: Allocator, key: *std.ArrayList(u8), text: []const u8) !void {
    for (text) |c| {
        if (std.ascii.isAlphanumeric(c)) try key.append(arena, std.ascii.toLower(c));
    }
}

/// Demangles a name of the Itanium C++ ABI that names a function, possibly in namespaces, whose
/// template arguments are types or integer literals. Any other symbol stays as it is.
fn demangle(arena: Allocator, symbol: []const u8) Allocator.Error!Name {
    const plain: Name = .{ .text = symbol, .base = symbol };
    if (!std.mem.startsWith(u8, symbol, "_Z")) return plain;
    var demangler: Demangler = .{ .arena = arena, .text = symbol, .index = "_Z".len };
    return demangler.name() catch |err| switch (err) {
        error.OutOfMemory => |e| e,
        error.Unsupported => plain,
    };
}

const Demangler = struct {
    arena: Allocator,
    text: []const u8,
    index: usize,

    const Error = error{ Unsupported, OutOfMemory };

    fn eat(d: *Demangler, byte: u8) bool {
        if (d.index >= d.text.len or d.text[d.index] != byte) return false;
        d.index += 1;
        return true;
    }

    fn name(d: *Demangler) Error!Name {
        var text: std.ArrayList(u8) = .empty;
        var base: []const u8 = "";
        var arguments: []const []const u8 = &.{};
        if (d.eat('N')) {
            while (!d.eat('E')) {
                if (d.eat('I')) {
                    arguments = try d.templateArguments();
                    continue;
                }
                if (base.len != 0) try text.print(d.arena, "{s}::", .{base});
                base = try d.sourceName();
                arguments = &.{};
            }
        } else {
            base = try d.sourceName();
            if (d.eat('I')) arguments = try d.templateArguments();
        }
        try text.appendSlice(d.arena, base);
        if (arguments.len != 0) {
            try text.append(d.arena, '<');
            for (arguments, 0..) |argument, index| {
                if (index != 0) try text.appendSlice(d.arena, ", ");
                try text.appendSlice(d.arena, argument);
            }
            try text.append(d.arena, '>');
        }
        return .{ .text = text.items, .base = base, .arguments = arguments };
    }

    fn sourceName(d: *Demangler) Error![]const u8 {
        const start = d.index;
        while (d.index < d.text.len and std.ascii.isDigit(d.text[d.index])) d.index += 1;
        const len = std.fmt.parseUnsigned(usize, d.text[start..d.index], 10) catch return error.Unsupported;
        if (d.text.len - d.index < len) return error.Unsupported;
        defer d.index += len;
        return d.text[d.index..][0..len];
    }

    fn templateArguments(d: *Demangler) Error![]const []const u8 {
        var arguments: std.ArrayList([]const u8) = .empty;
        while (!d.eat('E')) try arguments.append(d.arena, try d.templateArgument());
        return arguments.items;
    }

    fn templateArgument(d: *Demangler) Error![]const u8 {
        if (!d.eat('L')) return d.builtinOrSourceName();
        const boolean = d.eat('b');
        if (!boolean) _ = try d.builtinOrSourceName();
        const negative = d.eat('n');
        const start = d.index;
        while (d.index < d.text.len and std.ascii.isDigit(d.text[d.index])) d.index += 1;
        const digits = d.text[start..d.index];
        if (digits.len == 0 or !d.eat('E')) return error.Unsupported;
        if (boolean) return if (std.mem.eql(u8, digits, "0")) "false" else "true";
        return if (negative) try std.fmt.allocPrint(d.arena, "-{s}", .{digits}) else digits;
    }

    fn builtinOrSourceName(d: *Demangler) Error![]const u8 {
        if (d.index >= d.text.len) return error.Unsupported;
        if (std.ascii.isDigit(d.text[d.index])) return d.sourceName();
        d.index += 1;
        return switch (d.text[d.index - 1]) {
            'v' => "void",
            'b' => "bool",
            'c' => "char",
            'a' => "signed char",
            'h' => "unsigned char",
            's' => "short",
            't' => "unsigned short",
            'i' => "int",
            'j' => "unsigned int",
            'l' => "long",
            'm' => "unsigned long",
            'x' => "long long",
            'y' => "unsigned long long",
            'f' => "float",
            'd' => "double",
            else => error.Unsupported,
        };
    }
};

const Setup = struct {
    arch: []const u8,
    zig_mode: []const u8,
    nvcc_flags: []const u8,
};

fn writeReport(w: *Io.Writer, setup: Setup, comparisons: []const Comparison, skips: []const Skip) !void {
    try w.print(
        \\# Zig and nvcc kernels of the CUDA samples
        \\
        \\ptxas assembles each kernel for `{s}`: the PTX of the Zig port in the optimize mode
        \\`{s}`, and the PTX that nvcc compiles from the upstream sample with `{s}`. A cell gives
        \\the Zig value and then the nvcc value. A bold Zig value is greater than the nvcc value.
        \\SASS instructions counts the instructions other than NOP in the code section of the
        \\kernel. The section also holds a copy of each function that the kernel calls.
        \\
        \\## Summary
        \\
        \\| Value | Zig less | Same | Zig greater |
        \\|---|---:|---:|---:|
        \\
    , .{ setup.arch, setup.zig_mode, setup.nvcc_flags });
    var pair_count: usize = 0;
    var zig_instructions: u64 = 0;
    var nvcc_instructions: u64 = 0;
    var log_ratio_sum: f64 = 0;
    var log_ratio_count: usize = 0;
    for (comparisons) |comparison| {
        pair_count += comparison.pairs.len;
        for (comparison.pairs) |pair| {
            const zig = pair.zig.stats.instructions;
            const nvcc = pair.nvcc.stats.instructions;
            zig_instructions += zig;
            nvcc_instructions += nvcc;
            if (zig == 0 or nvcc == 0) continue;
            log_ratio_sum += @log(@as(f64, @floatFromInt(zig)) / @as(f64, @floatFromInt(nvcc)));
            log_ratio_count += 1;
        }
    }
    inline for (metrics) |metric| {
        var counts: [3]usize = @splat(0);
        for (comparisons) |comparison| {
            for (comparison.pairs) |pair| {
                const index: usize = switch (std.math.order(@field(pair.zig.stats, metric.field), @field(pair.nvcc.stats, metric.field))) {
                    .lt => 0,
                    .eq => 1,
                    .gt => 2,
                };
                counts[index] += 1;
            }
        }
        try w.print("| {s} | {d} | {d} | {d} |\n", .{ metric.title, counts[0], counts[1], counts[2] });
    }
    const mean_ratio = if (log_ratio_count == 0) 1 else @exp(log_ratio_sum / @as(f64, @floatFromInt(log_ratio_count)));
    try w.print(
        \\
        \\{d} kernel pairs in {d} samples. The Zig kernels have {d} SASS instructions and the nvcc
        \\kernels {d}. The geometric mean of the Zig count divided by the nvcc count is {d:.3}.
        \\
        \\Zig kernels that use local memory where the nvcc kernel uses none:
    , .{ pair_count, comparisons.len, zig_instructions, nvcc_instructions, mean_ratio });
    var local_count: usize = 0;
    for (comparisons) |comparison| {
        for (comparison.pairs) |pair| {
            if (!pair.zig.stats.usesLocalMemory() or pair.nvcc.stats.usesLocalMemory()) continue;
            try w.print("{s} `{s}`", .{ if (local_count == 0) "" else ",", pair.zig.symbol });
            local_count += 1;
        }
    }
    try w.print("{s}.\n\nZig kernels without an nvcc kernel:", .{if (local_count == 0) " none" else ""});
    var unpaired_count: usize = 0;
    for (comparisons) |comparison| {
        for (comparison.zig_unpaired) |function| {
            try w.print("{s} `{s}`{s}", .{
                if (unpaired_count == 0) "" else ",",
                function.symbol,
                if (isZigOnly(comparison.sample.name, function.symbol)) " (the upstream sample does not instantiate it)" else "",
            });
            unpaired_count += 1;
        }
    }
    try w.print("{s}.\n\nnvcc kernels without a Zig kernel:", .{if (unpaired_count == 0) " none" else ""});
    unpaired_count = 0;
    for (comparisons) |comparison| {
        for (comparison.nvcc_unpaired) |name| {
            try w.print("{s} `{s}`", .{ if (unpaired_count == 0) "" else ",", name.text });
            unpaired_count += 1;
        }
    }
    try w.print("{s}.\n\n", .{if (unpaired_count == 0) " none" else ""});
    for (skips) |skip| try w.print("The comparison leaves out {s}: {s}.\n\n", .{ skip.sample, skip.reason });

    for (comparisons) |comparison| {
        try w.print("## {s}\n\n| Kernel | nvcc kernel |", .{comparison.sample.name});
        inline for (metrics) |metric| try w.print(" {s} |", .{metric.title});
        try w.writeAll("\n|---|---|");
        inline for (metrics) |_| try w.writeAll("---:|");
        try w.writeAll("\n");
        for (comparison.pairs) |pair| {
            try w.print("| `{s}` | `{s}` |", .{ pair.zig.symbol, pair.nvcc_name.text });
            inline for (metrics) |metric| {
                const zig = @field(pair.zig.stats, metric.field);
                const nvcc = @field(pair.nvcc.stats, metric.field);
                if (zig > nvcc) {
                    try w.print(" **{d}** / {d} |", .{ zig, nvcc });
                } else {
                    try w.print(" {d} / {d} |", .{ zig, nvcc });
                }
            }
            try w.writeAll("\n");
        }
        try w.writeAll("\n");
        for ([_][]const Function{ comparison.sample.zig.items, comparison.sample.nvcc.items }, [_][]const u8{ "Zig", "nvcc" }) |functions, side| {
            for (functions) |function| {
                if (function.kernel) continue;
                try w.print("{s} function `{s}`: a stack frame of {d} bytes.\n\n", .{
                    side, function.symbol, function.stats.stack,
                });
            }
        }
    }
}
