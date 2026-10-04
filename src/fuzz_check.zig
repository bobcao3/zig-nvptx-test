const builtin = @import("builtin");
const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;
const Cuda = @import("cuda");

const usage =
    \\Usage: nvptx-fuzz-checker [--ptxas <path>] --arch
    \\       nvptx-fuzz-checker [--ptxas <path>] <module.ptx> <input.bin> <rows> <kernel>...
    \\
    \\Assembles the PTX module with ptxas for the first CUDA device and launches each kernel as
    \\`kernel(out: [*]u32, in: [*]const u64, first: [4]u64)` with one block of <rows> threads. The
    \\input buffer holds the bytes of <input.bin>, and `first` holds its first four words. Each kernel writes <rows> checksums that the GPU computes,
    \\followed by <rows> checksums that the compiler computed at compile time.
    \\
    \\The checker prints one line `<outcome>\t<kernel>\t<message>` per result. The outcome is
    \\pass, mismatch, ptxas_error, gpu_fault or timeout. The kernel of a ptxas_error, and of a
    \\gpu_fault that the driver reports while it loads the module, is `-`. The checker stops after
    \\a gpu_fault or timeout because the CUDA context cannot run another kernel after either.
    \\
    \\Exit status 0 means that every kernel passed, 2 that the checker printed another outcome,
    \\and 1 that the checker failed. `--arch` prints the PTX target of the device, such as `sm_90`.
    \\Without `--ptxas`, the checker runs $PTXAS, or else `ptxas` from PATH.
    \\
;

const Outcome = enum { pass, mismatch, ptxas_error, gpu_fault, timeout };

const Report = struct {
    io: Io,
    findings: u32 = 0,

    fn line(report: *Report, outcome: Outcome, kernel: []const u8, message: []const u8) !void {
        if (outcome != .pass) report.findings += 1;
        var buffer: [1024]u8 = undefined;
        var w: Io.Writer = .fixed(&buffer);
        w.print("{t}\t{s}\t", .{ outcome, kernel }) catch {};
        for (message[0..@min(message.len, 800)]) |c| {
            w.writeByte(if (c == '\n' or c == '\t' or c == '\r') ' ' else c) catch {};
        }
        w.writeByte('\n') catch {};
        try Io.File.stdout().writeStreamingAll(report.io, w.buffered());
    }

    fn exit(report: *const Report) noreturn {
        std.process.exit(if (report.findings == 0) 0 else 2);
    }
};

const poll_interval: Io.Duration = .fromMilliseconds(1);
const poll_limit = 20_000;
/// ptxas leaves the frames of recursive calls out of the stack size that it gives a kernel.
const stack_size = 16 * 1024;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    var args = try init.minimal.args.toSlice(arena);

    var ptxas: []const u8 = init.environ_map.get("PTXAS") orelse switch (builtin.os.tag) {
        .windows => "ptxas.exe",
        else => "ptxas",
    };
    if (args.len >= 3 and std.mem.eql(u8, args[1], "--ptxas")) {
        ptxas = args[2];
        args = args[2..];
    }

    const cuda = Cuda.Raw.load() catch |err| fatal("unable to load {s}: {t}", .{ Cuda.library_name, err });
    cuda.check(cuda.cuInit(0), "cuInit");
    var device: c_int = undefined;
    cuda.check(cuda.cuDeviceGet(&device, 0), "cuDeviceGet");
    var major: c_int = undefined;
    var minor: c_int = undefined;
    cuda.check(cuda.cuDeviceGetAttribute(&major, .compute_capability_major, device), "cuDeviceGetAttribute");
    cuda.check(cuda.cuDeviceGetAttribute(&minor, .compute_capability_minor, device), "cuDeviceGetAttribute");
    const arch = try std.fmt.allocPrint(arena, "sm_{d}{d}", .{ major, minor });

    if (args.len == 2 and std.mem.eql(u8, args[1], "--arch")) {
        try Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(arena, "{s}\n", .{arch}));
        return;
    }
    if (args.len < 5) fatal(usage, .{});
    const ptx_path = args[1];
    const input = try Io.Dir.cwd().readFileAlloc(io, args[2], arena, .unlimited);
    const rows = try std.fmt.parseUnsigned(u32, args[3], 10);
    const kernels = args[4..];
    var report: Report = .{ .io = io };

    const cubin_path = try std.fmt.allocPrint(arena, "{s}.cubin", .{ptx_path});
    const assembled = try std.process.run(arena, io, .{
        .argv = &.{ ptxas, try std.fmt.allocPrint(arena, "-arch={s}", .{arch}), ptx_path, "-o", cubin_path },
    });
    if (!assembled.term.success() or assembled.stderr.len != 0) {
        try report.line(.ptxas_error, "-", assembled.stderr);
        report.exit();
    }
    const image = try Io.Dir.cwd().readFileAlloc(io, cubin_path, arena, .unlimited);

    var context: ?*anyopaque = null;
    cuda.check(cuda.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    cuda.check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent");
    cuda.check(cuda.cuCtxSetLimit(.stack_size, stack_size), "cuCtxSetLimit");

    var log: [16 * 1024]u8 = @splat(0);
    const options = [_]Cuda.JitOption{ .error_log_buffer, .error_log_buffer_size_bytes };
    const option_values = [_]?*anyopaque{ &log, @ptrFromInt(log.len) };
    var module: ?*anyopaque = null;
    const loaded = cuda.cuModuleLoadDataEx(&module, image.ptr, options.len, &options, &option_values);
    if (loaded != 0) {
        const message = try std.fmt.allocPrint(arena, "cuModuleLoadDataEx: {s} {s}", .{
            cuda.errorName(loaded), std.mem.sliceTo(&log, 0),
        });
        try report.line(.gpu_fault, "-", message);
        report.exit();
    }

    const out_len = 2 * rows;
    var in: u64 = undefined;
    var out: u64 = undefined;
    cuda.check(cuda.cuMemAlloc_v2(&in, @max(input.len, 1)), "cuMemAlloc");
    cuda.check(cuda.cuMemAlloc_v2(&out, out_len * 4), "cuMemAlloc");
    if (input.len != 0) cuda.check(cuda.cuMemcpyHtoD_v2(in, input.ptr, input.len), "cuMemcpyHtoD");
    const words = try arena.alloc(u32, out_len);

    for (kernels) |name| {
        var function: ?*anyopaque = null;
        const found = cuda.cuModuleGetFunction(&function, module, name);
        if (found != 0) {
            const message = try std.fmt.allocPrint(arena, "cuModuleGetFunction: {s}", .{cuda.errorName(found)});
            try report.line(.gpu_fault, name, message);
            report.exit();
        }
        cuda.check(cuda.cuMemsetD32_v2(out, 0xdeadbeef, out_len), "cuMemsetD32");
        var first: [4]u64 = @splat(0);
        @memcpy(std.mem.sliceAsBytes(&first)[0..@min(input.len, 32)], input[0..@min(input.len, 32)]);
        const params = [_]*const anyopaque{ &out, &in, &first };
        var result = cuda.cuLaunchKernel(function, 1, 1, 1, rows, 1, 1, 0, null, &params, null);
        var polls: u32 = 0;
        if (result == 0) while (true) : (polls += 1) {
            result = cuda.cuStreamQuery(null);
            if (result != Cuda.error_not_ready) break;
            if (polls == poll_limit) {
                try report.line(.timeout, name, "");
                report.exit();
            }
            try io.sleep(poll_interval, .awake);
        };
        if (result != 0) {
            try report.line(.gpu_fault, name, cuda.errorName(result));
            report.exit();
        }
        cuda.check(cuda.cuMemcpyDtoH_v2(words.ptr, out, out_len * 4), "cuMemcpyDtoH");
        try compare(arena, &report, name, words[0..rows], words[rows..]);
    }
    report.exit();
}

fn compare(arena: std.mem.Allocator, report: *Report, kernel: []const u8, gpu: []const u32, expected: []const u32) !void {
    var differ: u32 = 0;
    var first: ?usize = null;
    for (gpu, expected, 0..) |g, e, r| {
        if (g == e) continue;
        differ += 1;
        if (first == null) first = r;
    }
    const r = first orelse return report.line(.pass, kernel, "");
    const message = try std.fmt.allocPrint(arena, "{d} rows differ; row {d}: gpu 0x{x} comptime 0x{x}", .{
        differ, r, gpu[r], expected[r],
    });
    try report.line(.mismatch, kernel, message);
}
