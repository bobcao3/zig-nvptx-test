const std = @import("std");
const contend_expected = @import("suites/atomics/contend_expected.zig");
const Sm = @import("src/Sm.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const host_is_windows = b.graph.host.result.os.tag == .windows;
    const compiler = b.option([]const u8, "zig", "The Zig compiler under test (default: the compiler that runs this build)") orelse b.graph.zig_exe;
    const ptxas = b.option([]const u8, "ptxas", "The ptxas that checks every PTX file (default: ptxas in PATH)") orelse
        b.findProgram(.{ .names = &.{if (host_is_windows) "ptxas.exe" else "ptxas"} });
    const only_targets = b.option([]const []const u8, "sm", "Run the suites for these targets, for example -Dsm=sm_90a (default: one baseline target per required feature set)");
    const behavior_targets = b.option([]const []const u8, "behavior-target", "A target that the behavior step compiles for (default: sm_80 and the exact target of the device)");
    const zig_src = b.option([]const u8, "zig-src", "The Zig checkout whose test/behavior files the behavior step runs (default: the checkout of the compiler under test)");
    const jobs = b.option(u32, "jobs", "How many behavior files the behavior step tests at once (default: the CPU count, at most 8)") orelse
        @as(u32, @intCast(std.math.clamp(std.Thread.getCpuCount() catch 4, 1, 8)));
    if (ptxas == null) std.log.warn("ptxas is not in PATH: the ptxas checks are skipped (pass -Dptxas=<path>)", .{});

    const installed = addTools(b, target, optimize);
    b.installArtifact(installed.executor);
    b.installArtifact(installed.launcher);
    b.installArtifact(installed.behavior);
    b.installArtifact(installed.fuzz_check);

    const tools = addTools(b, b.graph.host, optimize);

    const test_step = b.step("test", "Compile the GPU suites with the compiler under test, check them with ptxas and run them on the GPU");
    const behavior_step = b.step("behavior", "Run each behavior test file of -Dzig-src on the GPU and write zig-out/behavior.md");
    var suite: Suite = .{
        .b = b,
        .zig = compiler,
        .ptxas = ptxas,
        .only_targets = only_targets,
        .launcher = tools.launcher,
        .step = undefined,
        .inputs = compilerInputs(b, compiler),
    };
    const suites = .{
        .{ "kernels", "Run the kernel suite", addKernelSuite },
        .{ "int", "Run the integer and float suite", addIntSuite },
        .{ "regressions", "Run the regression suite", addRegressionsSuite },
        .{ "std", "Run the std.nvptx suite", addStdSuite },
        .{ "pointers", "Run the address space suite", addPointerSuite },
        .{ "atomics", "Run the atomics suite", addAtomicsSuite },
        .{ "unions", "Run the unions, enums and function pointers suite", addUnionSuite },
        .{ "vectors", "Run the vector suite", addVectorSuite },
        .{ "ptxas", "Run the suite of the forms that ptxas compiles wrongly", addPtxasSuite },
    };
    inline for (suites) |entry| {
        suite.step = b.step("test-" ++ entry[0], entry[1]);
        entry[2](&suite);
        test_step.dependOn(suite.step);
    }

    const source = zig_src orelse findZigSource(b, compiler) orelse {
        behavior_step.dependOn(&b.addFail(b.fmt(
            "the library directory of {s} has no test/behavior beside it: pass -Dzig-src=<path of a Zig checkout>",
            .{compiler},
        )).step);
        return;
    };
    const behavior = b.addRunArtifact(tools.behavior);
    behavior.has_side_effects = true;
    behavior.addArgs(&.{
        b.fmt("--zig-src={s}", .{source}),
        b.fmt("--zig={s}", .{compiler}),
        b.fmt("--jobs={d}", .{jobs}),
    });
    for (behavior_targets orelse &.{}) |target_name| behavior.addArg(b.fmt("--mcpu={s}", .{target_name}));
    if (ptxas) |path| behavior.addArg(b.fmt("--ptxas={s}", .{path}));
    behavior.addPrefixedArtifactArg("--executor=", tools.executor);
    behavior.addPrefixedFileArg("--test-runner=", b.path("src/test_runner.zig"));
    const table = behavior.addPrefixedOutputFileArg("--out=", "behavior.md");
    behavior.addPassthruArgs();
    behavior_step.dependOn(&b.addInstallFile(table, "behavior.md").step);
}

/// Returns the checkout that contains the library directory of `compiler`, or null if the
/// library directory has no `test/behavior` beside it.
fn compilerInputs(b: *std.Build, compiler: []const u8) []const std.Build.LazyPath {
    var inputs: std.ArrayList(std.Build.LazyPath) = .empty;
    inputs.append(b.allocator, .{ .cwd_relative = compiler }) catch @panic("OOM");
    const lib_dir = libDir(b, compiler) orelse return inputs.items;
    inputs.append(b.allocator, .{ .cwd_relative = b.pathJoin(&.{ lib_dir, "std", "nvptx.zig" }) }) catch @panic("OOM");
    const nvptx_dir = b.pathJoin(&.{ lib_dir, "std", "nvptx" });
    var dir = std.Io.Dir.cwd().openDir(b.graph.io, nvptx_dir, .{ .iterate = true }) catch return inputs.items;
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (it.next(b.graph.io) catch null) |entry| {
        if (entry.kind != .file) continue;
        inputs.append(b.allocator, .{ .cwd_relative = b.pathJoin(&.{ nvptx_dir, entry.name }) }) catch @panic("OOM");
    }
    return inputs.items;
}

fn findZigSource(b: *std.Build, compiler: []const u8) ?[]const u8 {
    const lib_dir = libDir(b, compiler) orelse return null;
    const root = std.fs.path.dirname(lib_dir) orelse return null;
    const behavior_dir = b.pathJoin(&.{ root, "test", "behavior" });
    std.Io.Dir.cwd().access(b.graph.io, behavior_dir, .{}) catch return null;
    return root;
}

/// Returns the library directory that `compiler` reports with `zig env`. A compiler built with
/// only the PTX backend lacks that command, so the search then walks up from the compiler file
/// the way the compiler itself does.
fn libDir(b: *std.Build, compiler: []const u8) ?[]const u8 {
    switch (b.runFallible(&.{ compiler, "env" }, .{ .stderr_behavior = .ignore })) {
        .success => |output| {
            const key = ".lib_dir = ";
            const start = (std.mem.find(u8, output, key) orelse return null) + key.len;
            const end = std.mem.findScalarPos(u8, output, start, '\n') orelse return null;
            const literal = std.mem.trimEnd(u8, output[start..end], ",");
            return std.zig.string_literal.parseAlloc(b.allocator, literal) catch null;
        },
        else => {},
    }
    var dir = std.fs.path.dirname(compiler);
    while (dir) |path| : (dir = std.fs.path.dirname(path)) {
        const candidate = b.pathJoin(&.{ path, "lib" });
        std.Io.Dir.cwd().access(b.graph.io, b.pathJoin(&.{ candidate, "std", "std.zig" }), .{}) catch continue;
        return candidate;
    }
    return null;
}

const Tools = struct {
    executor: *std.Build.Step.Compile,
    launcher: *std.Build.Step.Compile,
    behavior: *std.Build.Step.Compile,
    fuzz_check: *std.Build.Step.Compile,
};

fn addTools(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Tools {
    const cuda = b.createModule(.{
        .root_source_file = b.path("cuda.zig"),
        .target = target,
        .optimize = optimize,
        // `std.DynLib` uses `dlopen` only when the program links libc.
        .link_libc = target.result.os.tag == .linux,
    });
    const sm = b.createModule(.{ .root_source_file = b.path("src/Sm.zig"), .target = target, .optimize = optimize });
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "Sm", .module = sm },
    };
    return .{
        .executor = b.addExecutable(.{
            .name = "zig-nvptx-test-executor",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/executor.zig"),
                .target = target,
                .optimize = optimize,
                .imports = imports,
            }),
        }),
        .launcher = b.addExecutable(.{
            .name = "zig-nvptx-launch",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/launch.zig"),
                .target = target,
                .optimize = optimize,
                .imports = imports,
            }),
        }),
        .behavior = b.addExecutable(.{
            .name = "zig-nvptx-behavior",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/behavior.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "cuda", .module = cuda }},
            }),
        }),
        .fuzz_check = b.addExecutable(.{
            .name = "zig-nvptx-fuzz-check",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/fuzz_check.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "cuda", .module = cuda }},
                .link_libc = target.result.os.tag != .windows,
            }),
        }),
    };
}

const optimize_modes = [_][]const u8{ "ReleaseFast", "Debug", "ReleaseSafe" };

/// The build chooses the first compatible target unless the user supplies `-Dsm`.
const targets = [_][]const u8{
    "sm_75",   "sm_80",   "sm_86",   "sm_89",   "sm_90",   "sm_90a",
    "sm_100",  "sm_100a", "sm_100f", "sm_103a", "sm_110a", "sm_120",
    "sm_120a", "sm_120f", "sm_121",  "sm_121a", "sm_121f",
};

const Needs = struct {
    /// The smallest compute capability, as major * 10 + minor, that the test compiles for.
    min_capability: u32 = 80,
    /// When not empty, the only targets that the test compiles for.
    only: []const []const u8 = &.{},
    min_ptx: u16 = 0,
};

const needs_cluster: Needs = .{ .min_capability = 90 };
const needs_tma: Needs = .{ .min_capability = 90, .min_ptx = 80 };
const needs_wgmma: Needs = .{ .only = &.{"sm_90a"} };
const needs_tcgen05: Needs = .{ .only = &.{ "sm_100a", "sm_100f", "sm_103a", "sm_110a" } };
const needs_sm_75: Needs = .{ .min_capability = 75 };

const Suite = struct {
    b: *std.Build,
    zig: []const u8,
    ptxas: ?[]const u8,
    only_targets: ?[]const []const u8,
    launcher: *std.Build.Step.Compile,
    step: *std.Build.Step,
    /// The compiler under test and the `std.nvptx` sources it compiles with.
    inputs: []const std.Build.LazyPath,

    /// Returns the `-mcpu` values of the targets that satisfy `needs`.
    fn cpus(suite: *const Suite, needs: Needs) []const []const u8 {
        var result: std.ArrayList([]const u8) = .empty;
        for (targets) |target| {
            const sm = Sm.parse(target).?;
            if (sm.major * 10 + sm.minor < needs.min_capability) continue;
            if (needs.only.len != 0 and !contains(needs.only, target)) continue;
            if (suite.only_targets) |only| if (!contains(only, target)) continue;
            const query = std.Target.Query.parse(.{
                .arch_os_abi = "nvptx64-cuda",
                .cpu_features = target,
            }) catch unreachable;
            const cpu = suite.b.resolveTargetQuery(query).result.cpu;
            const ptx = std.nvptx.Gpu.fromCpu(cpu).ptx;
            const name = if (ptx < needs.min_ptx)
                suite.b.fmt("{s}+ptx{d}", .{ target, needs.min_ptx })
            else
                target;
            result.append(suite.b.allocator, name) catch @panic("OOM");
            if (suite.only_targets == null) break;
        }
        return result.items;
    }

    fn contains(list: []const []const u8, name: []const u8) bool {
        for (list) |item| if (std.mem.eql(u8, item, name)) return true;
        return false;
    }

    /// Compiles `file` with the compiler under test and checks the PTX with ptxas.
    fn compile(suite: *Suite, file: []const u8, cpu: []const u8, optimize: []const u8, ptxas_relocatable: bool) std.Build.LazyPath {
        const b = suite.b;
        const stem = std.fs.path.stem(file);
        const run = b.addSystemCommand(&.{ suite.zig, "build-exe", "-fno-llvm", "-target", "nvptx64-cuda" });
        run.setName(b.fmt("compile {s} {s} {s}", .{ file, cpu, optimize }));
        for (suite.inputs) |input| run.addFileInput(input);
        const dir = std.fs.path.dirname(file) orelse ".";
        for (listDir(b, dir)) |name| {
            if (std.mem.endsWith(u8, name, ".zig")) run.addFileInput(b.path(b.pathJoin(&.{ dir, name })));
        }
        run.addArgs(&.{ b.fmt("-mcpu={s}", .{cpu}), b.fmt("-O{s}", .{optimize}) });
        const ptx = run.addPrefixedOutputFileArg("-femit-bin=", b.fmt("{s}.ptx", .{stem}));
        run.addFileArg(b.path(file));
        run.expectExitCode(0);

        const ptxas = suite.ptxas orelse {
            suite.step.dependOn(&run.step);
            return ptx;
        };
        const check = b.addSystemCommand(&.{ptxas});
        check.setName(b.fmt("ptxas {s} {s} {s}", .{ file, cpu, optimize }));
        if (ptxas_relocatable) check.addArg("-c");
        check.addArg(b.fmt("-arch={s}", .{archOf(cpu)}));
        check.addFileArg(ptx);
        check.addArg("-o");
        _ = check.addOutputFileArg(b.fmt("{s}.cubin", .{stem}));
        check.expectExitCode(0);
        suite.step.dependOn(&check.step);
        return ptx;
    }

    /// Fails when ptxas gives a kernel of `ptx` a stack frame or spills one of its registers.
    fn expectRegisters(suite: *Suite, ptx: std.Build.LazyPath, file: []const u8, cpu: []const u8, optimize: []const u8) void {
        const b = suite.b;
        const ptxas = suite.ptxas orelse return;
        const check = b.addSystemCommand(&.{ ptxas, "--warn-on-local-memory-usage", "--warn-on-spills", "--warning-as-error" });
        check.setName(b.fmt("ptxas registers {s} {s} {s}", .{ file, cpu, optimize }));
        check.addArg(b.fmt("-arch={s}", .{archOf(cpu)}));
        check.addFileArg(ptx);
        check.addArg("-o");
        _ = check.addOutputFileArg(b.fmt("{s}.cubin", .{std.fs.path.stem(file)}));
        check.expectExitCode(0);
        suite.step.dependOn(&check.step);
    }

    fn archOf(cpu: []const u8) []const u8 {
        return cpu[0 .. std.mem.findScalar(u8, cpu, '+') orelse cpu.len];
    }

    fn launch(suite: *Suite, name: []const u8, ptx: std.Build.LazyPath) *std.Build.Step.Run {
        const b = suite.b;
        const run = b.addRunArtifact(suite.launcher);
        run.setName(b.fmt("gpu {s}", .{name}));
        run.has_side_effects = true;
        run.enableTestRunnerMode();
        run.addFileArg(ptx);
        suite.step.dependOn(&run.step);
        return run;
    }

    fn expect(run: *std.Build.Step.Run, kernel: []const u8, words: []const u8) void {
        run.addArgs(&.{ "expect", kernel, words });
    }

    fn expectFile(run: *std.Build.Step.Run, kernel: []const u8, words: std.Build.LazyPath) void {
        run.addArgs(&.{ "expect", kernel });
        run.addPrefixedFileArg("@", words);
    }
};

const calls_words = "9 7 5 1 3 2 100 532 240 0xfffffffc 2 0xffffffff 2.5 5 40 54";

const std_nvptx_words = "0x49249249 1 0 1 0 1 0xffffffff 0xffffffff 0xffffffff 0xffffffff 50 " ++
    "0xffffffff 496 11 0xffffffec 255 0 0xffffffff 0x3ff 2 1 1 0xffffffff 0xffffffff 0xffffffff " ++
    "1 1 1 1 1 1 1";

fn addKernelSuite(suite: *Suite) void {
    const b = suite.b;
    for (suite.cpus(.{})) |cpu| {
        for (optimize_modes) |optimize| {
            for ([_][]const u8{ "saxpy", "calls", "vec" }) |name| {
                const ptx = suite.compile(b.fmt("suites/kernels/{s}.zig", .{name}), cpu, optimize, false);
                const label = b.fmt("kernels {s} {s} {s}", .{ name, cpu, optimize });
                if (std.mem.eql(u8, name, "saxpy")) suite.launch(label, ptx).addArg("saxpy");
                if (std.mem.eql(u8, name, "calls")) Suite.expect(suite.launch(label, ptx), "calls", calls_words);
                if (std.mem.eql(u8, name, "vec")) Suite.expect(suite.launch(label, ptx), "vec", "40");
            }
            const std_nvptx = suite.compile("suites/kernels/std_nvptx.zig", cpu, optimize, false);
            suite.launch(b.fmt("kernels std_nvptx {s} {s}", .{ cpu, optimize }), std_nvptx).addArgs(&.{
                "contend",
                "stdNvptx",
                "2",
                "64",
                std_nvptx_words,
            });
        }
    }
    for (suite.cpus(needs_cluster)) |cpu| {
        for (optimize_modes) |optimize| _ = suite.compile("suites/kernels/options.zig", cpu, optimize, false);
    }
    for (suite.cpus(needs_tma)) |cpu| {
        const tma = suite.compile("suites/kernels/tma.zig", cpu, "ReleaseFast", false);
        suite.launch(b.fmt("kernels tma {s}", .{cpu}), tma).addArg("divmod");
        const tmaload = suite.compile("suites/kernels/tmaload.zig", cpu, "ReleaseFast", false);
        suite.launch(b.fmt("kernels tmaload {s}", .{cpu}), tmaload).addArg("tma");
    }
    for (suite.cpus(needs_wgmma)) |cpu| {
        _ = suite.compile("suites/kernels/gemm.zig", cpu, "ReleaseFast", false);
        _ = suite.compile("suites/kernels/wgmma.zig", cpu, "ReleaseFast", false);
    }
    for (suite.cpus(needs_tcgen05)) |cpu| _ = suite.compile("suites/kernels/tmem.zig", cpu, "ReleaseFast", false);
}

/// Compares every exported kernel of `<group>_kernels.zig` with `<group>_host.zig`, which runs
/// the same code on the host and prints the words that the kernel must store.
fn addIntSuite(suite: *Suite) void {
    addComparedGroups(suite, "int", &.{ "int", "float", "loops" }, .allowed);
}

fn addPointerSuite(suite: *Suite) void {
    addKernelChecks(suite, "pointers");
}

fn addStdSuite(suite: *Suite) void {
    addKernelChecks(suite, "std");
}

fn addVectorSuite(suite: *Suite) void {
    addKernelChecks(suite, "vectors");
}

fn addPtxasSuite(suite: *Suite) void {
    addKernelChecks(suite, "ptxas");
}

/// Runs every exported kernel of every file in `suites/<dir>/` on one work group of 128 work items.
/// A kernel stores 0 in `out[0]` when all of its checks pass. Otherwise it stores the line number
/// of a failed check.
fn addKernelChecks(suite: *Suite, dir: []const u8) void {
    const b = suite.b;
    for (listDir(b, b.fmt("suites/{s}", .{dir}))) |name| {
        if (!std.mem.endsWith(u8, name, ".zig")) continue;
        const file = b.fmt("suites/{s}/{s}", .{ dir, name });
        const kernels = exportedFunctions(b, file);
        const kernel_cpus = b.allocator.alloc([]const []const u8, kernels.len) catch @panic("OOM");
        var cpus: std.ArrayList([]const u8) = .empty;
        for (kernels, kernel_cpus) |kernel, *selected| {
            selected.* = suite.cpus(kernelNeeds(dir, name, kernel));
            for (selected.*) |cpu| {
                if (!Suite.contains(cpus.items, cpu)) cpus.append(b.allocator, cpu) catch @panic("OOM");
            }
        }
        for (cpus.items) |cpu| {
            for (optimize_modes) |optimize| {
                const ptx = suite.compile(file, cpu, optimize, false);
                for (kernels, kernel_cpus) |kernel, selected| {
                    if (!Suite.contains(selected, cpu)) continue;
                    const label = b.fmt("{s} {s} {s} {s} {s}", .{ dir, name, kernel, cpu, optimize });
                    suite.launch(label, ptx).addArgs(&.{ "contend", kernel, "1", "128", "0" });
                }
            }
        }
    }
}

fn kernelNeeds(dir: []const u8, file: []const u8, kernel: []const u8) Needs {
    if (!std.mem.eql(u8, dir, "std")) return .{};
    if (std.mem.eql(u8, file, "async.zig")) {
        if (Suite.contains(&.{ "mbarrier_wait_parity", "mbarrier_transactions" }, kernel)) return needs_tma;
        return .{ .min_ptx = 74 };
    }
    if (std.mem.eql(u8, file, "atomic.zig") and std.mem.eql(u8, kernel, "elect_sync")) return needs_tma;
    if (std.mem.eql(u8, file, "matrix.zig")) return .{ .min_ptx = 78 };
    if (std.mem.eql(u8, file, "math.zig")) return .{ .min_ptx = 76 };
    return .{};
}

fn addRegressionsSuite(suite: *Suite) void {
    addComparedGroups(suite, "regressions", &.{"regressions"}, .allowed);
    addComparedGroups(suite, "regressions", &.{ "locals", "mixed" }, .forbidden);
    for (suite.cpus(.{})) |cpu| _ = suite.compile("suites/regressions/large_parameter.zig", cpu, "ReleaseFast", false);
}

/// Whether the kernels of a group may keep variables in local memory.
const LocalMemory = enum { allowed, forbidden };

fn addComparedGroups(suite: *Suite, dir: []const u8, groups: []const []const u8, local_memory: LocalMemory) void {
    const b = suite.b;
    for (groups) |group| {
        const host = b.addExecutable(.{
            .name = b.fmt("{s}_host", .{group}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("suites/{s}/{s}_host.zig", .{ dir, group })),
                .target = b.graph.host,
                .optimize = .ReleaseFast,
            }),
        });
        const kernels_file = b.fmt("suites/{s}/{s}_kernels.zig", .{ dir, group });
        const names = exportedFunctions(b, kernels_file);
        const words = b.allocator.alloc(std.Build.LazyPath, names.len) catch @panic("OOM");
        for (names, words) |name, *word_file| {
            const run = b.addRunArtifact(host);
            run.addArg(name);
            word_file.* = run.captureStdOut(.{});
        }
        for (suite.cpus(.{})) |cpu| {
            for (optimize_modes) |optimize| {
                const ptx = suite.compile(kernels_file, cpu, optimize, false);
                if (local_memory == .forbidden) suite.expectRegisters(ptx, kernels_file, cpu, optimize);
                const run = suite.launch(b.fmt("{s} {s} {s}", .{ group, cpu, optimize }), ptx);
                for (names, words) |name, word_file| Suite.expectFile(run, name, word_file);
            }
        }
    }
}

/// Returns the names of the functions that `file` declares with `export fn`.
fn exportedFunctions(b: *std.Build, file: []const u8) []const []const u8 {
    b.dependOnFileContents(b.path(file));
    const path = b.root.join(b.allocator, file) catch @panic("OOM");
    const source = path.root_dir.handle.readFileAlloc(b.graph.io, path.sub_path, b.allocator, .unlimited) catch |err|
        std.process.fatal("unable to read {s}: {t}", .{ file, err });
    var names: std.ArrayList([]const u8) = .empty;
    var rest = source;
    while (std.mem.find(u8, rest, "export fn ")) |start| {
        rest = rest[start + "export fn ".len ..];
        var end: usize = 0;
        while (end < rest.len and (std.ascii.isLower(rest[end]) or std.ascii.isDigit(rest[end]) or rest[end] == '_')) end += 1;
        names.append(b.allocator, rest[0..end]) catch @panic("OOM");
    }
    return names.items;
}

fn addAtomicsSuite(suite: *Suite) void {
    const b = suite.b;
    const files = [_][]const u8{ "atoms", "atoms2", "contend", "mem", "orders" };
    for (suite.cpus(needs_sm_75)) |cpu| {
        var ptx: [files.len]std.Build.LazyPath = undefined;
        for (files, &ptx) |file, *file_ptx| {
            file_ptx.* = suite.compile(b.fmt("suites/atomics/{s}.zig", .{file}), cpu, "ReleaseFast", false);
        }
        const atoms = suite.launch(b.fmt("atomics atoms {s}", .{cpu}), ptx[0]);
        for ([_][]const u8{ "rmwGlobal", "rmwShared", "rmwLocal", "rmwGeneric" }) |kernel| Suite.expect(atoms, kernel, "0");
        Suite.expect(atoms, "localGeneric", "110 16 9");
        const atoms2 = suite.launch(b.fmt("atomics atoms2 {s}", .{cpu}), ptx[1]);
        for ([_][]const u8{ "casGlobal", "casShared", "lsGlobal", "lsShared", "casGeneric", "casLocal" }) |kernel| {
            Suite.expect(atoms2, kernel, "0");
        }
        Suite.expect(suite.launch(b.fmt("atomics mem {s}", .{cpu}), ptx[3]), "mem", "3229614080 16120 85480 89080 5973 5629680 239 19");

        const contend = suite.launch(b.fmt("atomics contend {s}", .{cpu}), ptx[2]);
        for (contend_expected.launches(b.allocator)) |launch| {
            contend.addArgs(&.{
                "contend",
                launch.kernel,
                b.fmt("{d}", .{launch.grid}),
                b.fmt("{d}", .{launch.block}),
                contend_expected.format(b.allocator, launch),
            });
        }
    }
}

/// Each test file runs its code through `harness.zig` as the kernel `kernel` on the GPU and as
/// `H.hostMain` on the host, which prints the words that the kernel must store.
fn addUnionSuite(suite: *Suite) void {
    const b = suite.b;
    const tests = [_][]const u8{
        "union_tagged", "union_layouts", "union_const",  "union_enum",
        "enum_names",   "enum_checks",   "field_parent", "fn_pointers",
        "small_params", "no_address",    "extern_fn",
    };
    for (tests) |name| {
        const file = b.fmt("suites/unions/{s}.zig", .{name});
        const defs_file = b.fmt("suites/unions/{s}.defs.zig", .{name});
        const has_defs = std.mem.eql(u8, name, "extern_fn");
        const words = hostWords(b, file, if (has_defs) defs_file else null);
        for (suite.cpus(.{})) |cpu| {
            for ([_][]const u8{ "ReleaseFast", "ReleaseSmall", "ReleaseSafe", "Debug" }) |optimize| {
                const defs_ptx = if (has_defs) suite.compile(defs_file, cpu, optimize, true) else null;
                const ptx = suite.compile(file, cpu, optimize, has_defs);
                const run = suite.launch(b.fmt("unions {s} {s} {s}", .{ name, cpu, optimize }), ptx);
                if (defs_ptx) |defs| run.addPrefixedFileArg("--link=", defs);
                Suite.expectFile(run, "kernel", words);
            }
        }
    }
    for ([_][]const u8{ "errint_trap", "enum_trap", "error_trap" }) |name| {
        for (suite.cpus(.{})) |cpu| {
            for ([_][]const u8{ "ReleaseSafe", "Debug" }) |optimize| {
                const ptx = suite.compile(b.fmt("suites/unions/{s}.zig", .{name}), cpu, optimize, false);
                suite.launch(b.fmt("unions {s} {s} {s}", .{ name, cpu, optimize }), ptx).addArgs(&.{ "trap", "kernel" });
            }
        }
    }
}

fn hostWords(b: *std.Build, file: []const u8, defs_file: ?[]const u8) std.Build.LazyPath {
    const main = b.addWriteFiles().add("host.zig", "pub fn main() void {\n    @import(\"test\").H.hostMain();\n}\n");
    const root_module = b.createModule(.{
        .root_source_file = main,
        .target = b.graph.host,
        .optimize = .Debug,
    });
    root_module.addImport("test", b.createModule(.{
        .root_source_file = b.path(file),
        .target = b.graph.host,
        .optimize = .Debug,
    }));
    if (defs_file) |defs| root_module.addObject(b.addObject(.{
        .name = std.fs.path.stem(defs),
        .root_module = b.createModule(.{
            .root_source_file = b.path(defs),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    }));
    const exe = b.addExecutable(.{ .name = std.fs.path.stem(file), .root_module = root_module });
    return b.addRunArtifact(exe).captureStdErr(.{});
}

fn listDir(b: *std.Build, sub_path: []const u8) []const []const u8 {
    const io = b.graph.io;
    b.dependOnDirectoryContents(b.path(sub_path));
    const path = b.root.join(b.allocator, sub_path) catch @panic("OOM");
    var dir = path.root_dir.handle.openDir(io, path.sub_path, .{ .iterate = true }) catch |err|
        std.process.fatal("unable to open {s}: {t}", .{ sub_path, err });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var iterator = dir.iterate();
    while (iterator.next(io) catch |err| std.process.fatal("unable to list {s}: {t}", .{ sub_path, err })) |entry| {
        if (entry.kind == .file) names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }
    std.mem.sortUnstable([]const u8, names.items, {}, lessThan);
    return names.items;
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}
