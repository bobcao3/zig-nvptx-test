const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const ptxas = b.option([]const u8, "ptxas", "The CUDA assembler") orelse "ptxas";
    const device_target = b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "nvptx64-cuda",
        .cpu_features = "sm_80+ptx70",
    }) catch unreachable);

    const scene = b.createModule(.{ .root_source_file = b.path("src/scene.zig") });
    const kernel = b.addExecutable(.{
        .name = "render",
        .use_llvm = false,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/render.zig"),
            .target = device_target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "scene", .module = scene }},
        }),
    });
    const ptx = kernel.getEmittedBin();
    const assemble = b.addSystemCommand(&.{ ptxas, "-arch=sm_80" });
    assemble.addFileArg(ptx);
    assemble.addArg("-o");
    _ = assemble.addOutputFileArg("render.cubin");

    const cuda = b.createModule(.{
        .root_source_file = b.path("../../cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = target.result.os.tag == .linux,
    });
    const host = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "scene", .module = scene },
        },
    });
    host.addAnonymousImport("render.ptx", .{ .root_source_file = ptx });
    const exe = b.addExecutable(.{ .name = "ray_tracing_in_one_weekend", .root_module = host });
    exe.step.dependOn(&assemble.step);
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    b.step("run", "Render the final scene to image.png").dependOn(&run.step);

    const png_test = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/png.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "scene", .module = scene }},
    }) });
    b.step("test", "Run the tests of the PNG encoder").dependOn(&b.addRunArtifact(png_test).step);
}
