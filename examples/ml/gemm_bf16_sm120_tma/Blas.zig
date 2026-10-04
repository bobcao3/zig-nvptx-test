const std = @import("std");
const cuda = @import("cuda");

const Blas = @This();

library: std.DynLib,
handle: ?*anyopaque,
destroy: *const fn (?*anyopaque) callconv(.c) c_int,
sgemm: *const fn (?*anyopaque, c_int, c_int, c_int, c_int, c_int, *const f32, cuda.DevicePointer, c_int, cuda.DevicePointer, c_int, *const f32, cuda.DevicePointer, c_int) callconv(.c) c_int,

pub fn open(stream: cuda.Stream) !Blas {
    var library = try std.DynLib.open("libcublas.so");
    errdefer library.close();
    const create = library.lookup(*const fn (*?*anyopaque) callconv(.c) c_int, "cublasCreate_v2") orelse return error.MissingBlasSymbol;
    const destroy = library.lookup(@FieldType(Blas, "destroy"), "cublasDestroy_v2") orelse return error.MissingBlasSymbol;
    const set_stream = library.lookup(*const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int, "cublasSetStream_v2") orelse return error.MissingBlasSymbol;
    const set_math = library.lookup(*const fn (?*anyopaque, c_int) callconv(.c) c_int, "cublasSetMathMode") orelse return error.MissingBlasSymbol;
    var handle: ?*anyopaque = null;
    try check(create(&handle));
    errdefer _ = destroy(handle);
    try check(set_stream(handle, stream.handle));
    try check(set_math(handle, 2));
    return .{
        .library = library,
        .handle = handle,
        .destroy = destroy,
        .sgemm = library.lookup(@FieldType(Blas, "sgemm"), "cublasSgemm_v2") orelse return error.MissingBlasSymbol,
    };
}

pub fn close(blas: *Blas) void {
    _ = blas.destroy(blas.handle);
    blas.library.close();
}

pub fn reference(blas: Blas, a: cuda.Buffer(f32), b: cuda.Buffer(f32), c: cuda.Buffer(f32), m: u32, n: u32, k: u32) !void {
    const alpha: f32 = 1;
    const beta: f32 = 0;
    try check(blas.sgemm(blas.handle, 1, 0, @intCast(n), @intCast(m), @intCast(k), &alpha, b.address, @intCast(k), a.address, @intCast(k), &beta, c.address, @intCast(n)));
}

fn check(status: c_int) !void {
    if (status != 0) {
        std.log.err("cuBLAS failed with status {d}", .{status});
        return error.BlasFailed;
    }
}
