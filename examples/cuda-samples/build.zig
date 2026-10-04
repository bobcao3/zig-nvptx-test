//! Builds every sample folder `<category>/<sample>/` that contains `<sample>.zig`. The host
//! program `<sample>.zig` embeds the PTX of each `*_kernel.zig` file of the folder as
//! `@embedFile("<file stem>.ptx")`. The install step mirrors the sample folders, as the build tree
//! of the CMake build of cuda-samples does: `bin/<category>/<sample>/` holds the executable and
//! the PTX of its kernels.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const kernel_optimize = b.option(std.builtin.Optimize, "kernel-optimize", "The optimization mode of the kernels") orelse .fast;
    // nvcc writes the PTX version of its own toolkit; sm_80 alone would mean PTX 7.0.
    const gpu = b.option([]const u8, "gpu", "The GPU architecture and PTX version that the PTX code targets") orelse "sm_80+ptx78";
    const cuda_bin = b.option([]const u8, "cuda-bin", "The bin directory of the CUDA toolkit whose nvcc, ptxas and nvdisasm compare-nvcc runs (default: the directory of nvcc in PATH)");

    const device_query = std.Target.Query.parse(.{ .arch_os_abi = "nvptx64-cuda", .cpu_features = gpu }) catch |err|
        std.process.fatal("invalid -Dgpu={s}: {t}", .{ gpu, err });
    const device_target = b.resolveTargetQuery(device_query);
    const cuda = b.createModule(.{
        .root_source_file = b.path("../../cuda.zig"),
        .target = target,
        .optimize = optimize,
        // `std.DynLib` uses `dlopen` only when the program links libc.
        .link_libc = target.result.os.tag == .linux,
    });

    const cuda_test = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("../../cuda_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    }) });
    const test_step = b.step("test", "Run the tests of the cuda module");
    test_step.dependOn(&b.addRunArtifact(cuda_test).step);
    const header_step = b.step("test-cuda-header", "Check the CUDA bindings against cuda.h");
    if (cudaHeader(b)) |header| {
        const header_test = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("../../cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = target.result.os.tag == .linux,
            .imports = &.{.{
                .name = "cuda_h",
                .module = b.addTranslateC(.{
                    .root_source_file = .{ .cwd_relative = header },
                    .target = target,
                    .optimize = optimize,
                }).createModule(),
            }},
        }) });
        const run_header_test = b.addRunArtifact(header_test);
        test_step.dependOn(&run_header_test.step);
        header_step.dependOn(&run_header_test.step);
    } else {
        header_step.dependOn(&b.addFail("cuda.h was not found; pass -Dcuda-include=<toolkit include directory>").step);
    }

    const wmma = b.createModule(.{
        .root_source_file = b.path("../../device/wmma.zig"),
        .target = device_target,
        .optimize = kernel_optimize,
    });
    const run_all = b.step("run", "Run every sample");
    var previous_run: ?*std.Build.Step = null;
    var samples: std.ArrayList(Sample) = .empty;
    for (listDir(b, ".", .directory)) |category| {
        if (category.len < 2 or !std.ascii.isDigit(category[0]) or category[1] != '_') continue;
        for (listDir(b, category, .directory)) |name| {
            const dir = b.pathJoin(&.{ category, name });
            const files = listDir(b, dir, .file);
            const host_file = b.fmt("{s}.zig", .{name});
            if (!contains(files, host_file)) continue;

            const build_sample = b.step(b.fmt("build-{s}", .{name}), b.fmt("Build {s}", .{dir}));
            const host = b.createModule(.{
                .root_source_file = b.path(b.pathJoin(&.{ dir, host_file })),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "cuda", .module = cuda }},
            });
            var ptx_files: std.ArrayList(std.Build.LazyPath) = .empty;
            for (files) |file| {
                if (!std.mem.endsWith(u8, file, "_kernel.zig")) continue;
                const stem = file[0 .. file.len - ".zig".len];
                const kernel = b.addExecutable(.{
                    .name = stem,
                    .use_llvm = false,
                    .root_module = b.createModule(.{
                        .root_source_file = b.path(b.pathJoin(&.{ dir, file })),
                        .target = device_target,
                        .optimize = kernel_optimize,
                        .imports = &.{.{ .name = "wmma", .module = wmma }},
                    }),
                });
                const ptx = kernel.getEmittedBin();
                ptx_files.append(b.allocator, ptx) catch @panic("OOM");
                host.addAnonymousImport(b.fmt("{s}.ptx", .{stem}), .{ .root_source_file = ptx });
                const install_ptx = b.addInstallFile(ptx, b.pathJoin(&.{ "bin", dir, b.fmt("{s}.ptx", .{stem}) }));
                b.getInstallStep().dependOn(&install_ptx.step);
                build_sample.dependOn(&install_ptx.step);
            }

            const exe = b.addExecutable(.{ .name = name, .root_module = host });
            const install_exe = b.addInstallArtifact(exe, .{
                .dest_dir = .{ .override = .{ .custom = b.pathJoin(&.{ "bin", dir }) } },
            });
            b.getInstallStep().dependOn(&install_exe.step);
            build_sample.dependOn(&install_exe.step);

            const run = b.addRunArtifact(exe);
            run.addPassthruArgs();
            b.step(b.fmt("run-{s}", .{name}), b.fmt("Run {s}", .{dir})).dependOn(&run.step);

            // The samples of `run` take turns on the GPU, so that their output and timings do
            // not interleave.
            const run_in_turn = b.addRunArtifact(exe);
            if (previous_run) |previous| run_in_turn.step.dependOn(previous);
            previous_run = &run_in_turn.step;
            run_all.dependOn(&run_in_turn.step);
            samples.append(b.allocator, .{ .dir = dir, .ptx = ptx_files.items }) catch @panic("OOM");
        }
    }

    const compare_step = b.step("compare-nvcc", "Compare the kernels with nvcc's build of the upstream samples and write zig-out/nvcc-compare.md");
    if (b.lazyDependency("upstream", .{})) |upstream| {
        const nvcc: std.Build.LazyPath = if (cuda_bin) |dir|
            .{ .cwd_relative = b.pathJoin(&.{ dir, b.fmt("nvcc{s}", .{b.graph.host.result.exeFileExt()}) }) }
        else
            b.findProgramLazy(.{ .names = &.{"nvcc"} });
        addCompareNvcc(b, compare_step, .{
            .upstream = upstream,
            .nvcc = nvcc,
            .arch = device_target.result.cpu.model.name,
            .kernel_optimize = kernel_optimize,
            .host_optimize = optimize,
        }, samples.items);
    }
}

const Sample = struct {
    dir: []const u8,
    ptx: []const std.Build.LazyPath,
};

/// nvcc compiles each `.cu` file of an upstream sample with these flags, an `-arch` option and
/// include directories.
const nvcc_flags = [_][]const u8{ "-ptx", "-O3", "-std=c++17", "--extended-lambda" };

/// The upstream samples that nvcc does not compile without more system headers.
const nvcc_skipped = [_]struct { []const u8, []const u8 }{
    .{ "5_Domain_Specific/nbody", "bodysystemcuda.cu includes the OpenGL headers" },
};

const Compare = struct {
    upstream: *std.Build.Dependency,
    nvcc: std.Build.LazyPath,
    arch: []const u8,
    kernel_optimize: std.builtin.Optimize,
    host_optimize: std.builtin.Optimize,
};

fn addCompareNvcc(b: *std.Build, step: *std.Build.Step, compare: Compare, samples: []const Sample) void {
    const exe_ext = b.graph.host.result.exeFileExt();
    const bin = compare.nvcc.dirname();
    const ptxas = bin.path(b, b.fmt("ptxas{s}", .{exe_ext}));
    const nvdisasm = bin.path(b, b.fmt("nvdisasm{s}", .{exe_ext}));
    const tool = b.addExecutable(.{
        .name = "nvcc-compare",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nvcc_compare.zig"),
            .target = b.graph.host,
            .optimize = compare.host_optimize,
        }),
    });
    const report = b.addRunArtifact(tool);
    const report_file = report.addPrefixedOutputFileArg("--out=", "nvcc-compare.md");
    const check = b.addRunArtifact(tool);
    check.addArg("--check");
    const runs = [_]*std.Build.Step.Run{ report, check };
    const arch_number = compare.arch["sm_".len..];
    for (runs) |run| run.addArgs(&.{
        b.fmt("--arch={s}", .{compare.arch}),
        b.fmt("--zig-mode={t}", .{compare.kernel_optimize}),
        b.fmt("--nvcc-flags={s} -arch=compute_{s}", .{ std.mem.join(b.allocator, " ", &nvcc_flags) catch @panic("OOM"), arch_number }),
    });
    for (nvcc_skipped) |skip| {
        for (runs) |run| run.addArgs(&.{ "--skip", skip[0], skip[1] });
    }
    sample: for (samples) |sample| {
        for (nvcc_skipped) |skip| {
            if (std.mem.eql(u8, skip[0], sample.dir)) continue :sample;
        }
        const sample_dir = b.pathJoin(&.{ "cpp", sample.dir });
        const sources = upstreamSources(b, compare.upstream, sample_dir);
        for (runs) |run| run.addArgs(&.{ "--sample", sample.dir });
        for (sample.ptx) |ptx| {
            const module = assemble(b, ptxas, nvdisasm, compare.arch, ptx, sample.dir);
            for (runs) |run| addModule(run, "--zig", module);
        }
        for (sources) |source| {
            const nvcc = std.Build.Step.Run.create(b, b.fmt("nvcc {s}", .{source}));
            nvcc.addFileArg(compare.nvcc);
            nvcc.addArgs(&nvcc_flags);
            nvcc.addArg(b.fmt("-arch=compute_{s}", .{arch_number}));
            nvcc.addPrefixedDirectoryArg("-I", compare.upstream.path("Common"));
            nvcc.addPrefixedDirectoryArg("-I", compare.upstream.path(sample_dir));
            const inc = b.pathJoin(&.{ sample_dir, "inc" });
            if (upstreamHas(b, compare.upstream, inc)) nvcc.addPrefixedDirectoryArg("-I", compare.upstream.path(inc));
            nvcc.addArg("-o");
            const ptx = nvcc.addOutputFileArg(b.fmt("{s}.ptx", .{std.fs.path.stem(source)}));
            nvcc.addFileArg(compare.upstream.path(source));
            const module = assemble(b, ptxas, nvdisasm, compare.arch, ptx, source);
            for (runs) |run| addModule(run, "--nvcc", module);
        }
    }
    const install_report = b.addInstallFile(report_file, "nvcc-compare.md");
    check.step.dependOn(&install_report.step);
    step.dependOn(&check.step);
}

const Module = struct {
    info: std.Build.LazyPath,
    sass: std.Build.LazyPath,
};

/// Assembles `ptx` with `ptxas -v` and disassembles the result with `nvdisasm`.
fn assemble(b: *std.Build, ptxas: std.Build.LazyPath, nvdisasm: std.Build.LazyPath, arch: []const u8, ptx: std.Build.LazyPath, label: []const u8) Module {
    const assembler = std.Build.Step.Run.create(b, b.fmt("ptxas {s}", .{label}));
    assembler.addFileArg(ptxas);
    assembler.addArgs(&.{ b.fmt("-arch={s}", .{arch}), "-v", "-o" });
    const cubin = assembler.addOutputFileArg("module.cubin");
    assembler.addFileArg(ptx);
    const disassembler = std.Build.Step.Run.create(b, b.fmt("nvdisasm {s}", .{label}));
    disassembler.addFileArg(nvdisasm);
    disassembler.addArg("-c");
    disassembler.addFileArg(cubin);
    return .{
        .info = assembler.captureStdErr(.{ .basename = "ptxas.txt" }),
        .sass = disassembler.captureStdOut(.{ .basename = "sass.txt" }),
    };
}

fn addModule(run: *std.Build.Step.Run, side: []const u8, module: Module) void {
    run.addArg(side);
    run.addFileArg(module.info);
    run.addFileArg(module.sass);
}

/// Returns the sorted paths of the `.cu` files under `sub_path` of the upstream samples.
fn upstreamSources(b: *std.Build, upstream: *std.Build.Dependency, sub_path: []const u8) []const []const u8 {
    const io = b.graph.io;
    const path = upstream.builder.root.join(b.allocator, sub_path) catch @panic("OOM");
    var dir = path.root_dir.handle.openDir(io, path.sub_path, .{ .iterate = true }) catch |err|
        std.process.fatal("unable to open {s} of the upstream samples: {t}", .{ sub_path, err });
    defer dir.close(io);
    var walker = dir.walk(b.allocator) catch @panic("OOM");
    defer walker.deinit();
    var sources: std.ArrayList([]const u8) = .empty;
    while (walker.next(io) catch |err| std.process.fatal("unable to list {s} of the upstream samples: {t}", .{ sub_path, err })) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".cu")) continue;
        sources.append(b.allocator, b.pathJoin(&.{ sub_path, entry.path })) catch @panic("OOM");
    }
    std.mem.sortUnstable([]const u8, sources.items, {}, lessThan);
    return sources.items;
}

fn upstreamHas(b: *std.Build, upstream: *std.Build.Dependency, sub_path: []const u8) bool {
    const path = upstream.builder.root.join(b.allocator, sub_path) catch @panic("OOM");
    path.root_dir.handle.access(b.graph.io, path.sub_path, .{}) catch return false;
    return true;
}

/// Returns the sorted names of the entries of `kind` in the directory `sub_path` of the build root.
fn listDir(b: *std.Build, sub_path: []const u8, kind: std.Io.File.Kind) []const []const u8 {
    const io = b.graph.io;
    b.dependOnDirectoryContents(b.path(sub_path));
    const path = b.root.join(b.allocator, sub_path) catch @panic("OOM");
    var dir = path.root_dir.handle.openDir(io, path.sub_path, .{ .iterate = true }) catch |err|
        std.process.fatal("unable to open {s}: {t}", .{ sub_path, err });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var iterator = dir.iterate();
    while (iterator.next(io) catch |err| std.process.fatal("unable to list {s}: {t}", .{ sub_path, err })) |entry| {
        if (entry.kind != kind or entry.name[0] == '.') continue;
        names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }
    std.mem.sortUnstable([]const u8, names.items, {}, lessThan);
    return names.items;
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |entry| {
        if (std.mem.eql(u8, entry, name)) return true;
    }
    return false;
}

/// Finds `cuda.h` of an installed CUDA toolkit: `-Dcuda-include`, then `$CUDA_PATH/include`, then
/// `/usr/local/cuda/include`. The test step checks the values and layouts of the `cuda` module
/// against the header when one is found.
fn cudaHeader(b: *std.Build) ?[]const u8 {
    const io = b.graph.io;
    const candidates = [_]?[]const u8{
        b.option([]const u8, "cuda-include", "The include directory of a CUDA toolkit"),
        if (b.graph.environ_map.get("CUDA_PATH")) |root| b.pathJoin(&.{ root, "include" }) else null,
        "/usr/local/cuda/include",
    };
    for (candidates) |candidate| {
        const dir = candidate orelse continue;
        const header = b.pathJoin(&.{ dir, "cuda.h" });
        std.Io.Dir.accessAbsolute(io, header, .{}) catch continue;
        return header;
    }
    return null;
}
