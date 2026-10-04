const std = @import("std");

const Example = struct {
    name: []const u8,
    kernel_stem: []const u8,
    gpu: []const u8 = "sm_80+ptx70",
    architectures: []const []const u8 = &.{ "sm_80", "sm_86", "sm_89" },
};

const examples = [_]Example{
    .{
        .name = "gemm_bf16_sm120_tma",
        .kernel_stem = "gemm_kernel",
        .gpu = "sm_120+ptx86",
        .architectures = &.{"sm_120"},
    },
    .{ .name = "gemm_bf16_sm80_cp_async", .kernel_stem = "gemm_kernel" },
    .{ .name = "flash_attn_sm80", .kernel_stem = "attention_kernel" },
    .{ .name = "rms_norm_fwd_sm80", .kernel_stem = "rms_norm_kernel" },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .fast });
    const ptxas = b.option([]const u8, "ptxas", "The CUDA assembler") orelse "ptxas";
    const cuda = b.createModule(.{
        .root_source_file = b.path("../../cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = target.result.os.tag == .linux,
    });
    const table = b.createModule(.{
        .root_source_file = b.path("TerminalTableRenderer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "TerminalTableRenderer", .module = table },
    };

    for (examples) |example| {
        const build_example = b.step(b.fmt("build-{s}", .{example.name}), b.fmt("Build {s}", .{example.name}));
        b.getInstallStep().dependOn(build_example);
        const install_dir = b.pathJoin(&.{ "bin", example.name });
        const device_target = b.resolveTargetQuery(std.Target.Query.parse(.{
            .arch_os_abi = "nvptx64-cuda",
            .cpu_features = example.gpu,
        }) catch unreachable);
        const kernel = b.addExecutable(.{
            .name = example.kernel_stem,
            .use_llvm = false,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("{s}/{s}.zig", .{ example.name, example.kernel_stem })),
                .target = device_target,
                .optimize = .fast,
            }),
        });
        const ptx = kernel.getEmittedBin();
        const ptx_name = b.fmt("{s}.ptx", .{example.kernel_stem});
        build_example.dependOn(&b.addInstallFile(ptx, b.pathJoin(&.{ install_dir, ptx_name })).step);

        const host = b.createModule(.{
            .root_source_file = b.path(b.fmt("{s}/main.zig", .{example.name})),
            .target = target,
            .optimize = optimize,
            .imports = imports,
        });
        host.addAnonymousImport(ptx_name, .{ .root_source_file = ptx });
        const exe = b.addExecutable(.{ .name = example.name, .root_module = host });
        build_example.dependOn(&b.addInstallArtifact(exe, .{
            .dest_dir = .{ .override = .{ .custom = install_dir } },
        }).step);

        for (example.architectures) |arch| {
            const assemble = b.addSystemCommand(&.{ ptxas, b.fmt("-arch={s}", .{arch}) });
            assemble.addFileArg(ptx);
            assemble.addArg("-o");
            const cubin_name = b.fmt("{s}.{s}.cubin", .{ example.kernel_stem, arch });
            const cubin = assemble.addOutputFileArg(cubin_name);
            build_example.dependOn(&b.addInstallFile(cubin, b.pathJoin(&.{ install_dir, cubin_name })).step);
        }
    }
}
