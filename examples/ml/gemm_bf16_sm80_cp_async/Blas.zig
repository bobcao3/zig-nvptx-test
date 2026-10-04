const std = @import("std");
const cuda = @import("cuda");

const Blas = @This();

library: std.DynLib,
handle: ?*anyopaque,
destroy: *const fn (?*anyopaque) callconv(.c) c_int,
sgemm: *const fn (?*anyopaque, c_int, c_int, c_int, c_int, c_int, *const f32, cuda.DevicePointer, c_int, cuda.DevicePointer, c_int, *const f32, cuda.DevicePointer, c_int) callconv(.c) c_int,
gemm_ex: *const fn (
    handle: ?*anyopaque,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: cuda.DevicePointer,
    a_type: c_int,
    lda: c_int,
    b: cuda.DevicePointer,
    b_type: c_int,
    ldb: c_int,
    beta: *const f32,
    c: cuda.DevicePointer,
    c_type: c_int,
    ldc: c_int,
    compute_type: c_int,
    algorithm: c_int,
) callconv(.c) c_int,

/// The values of `cublasMath_t`.
pub const Math = enum(c_int) { default = 0, pedantic = 2 };

const operation_n = 0;
const operation_t = 1;
const data_type_f32 = 0;
const data_type_bf16 = 14;
const compute_type_f32 = 68;
const algorithm_default = -1;

pub fn open(stream: cuda.Stream, math: Math) !Blas {
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
    try check(set_math(handle, @backingInt(math)));
    return .{
        .library = library,
        .handle = handle,
        .destroy = destroy,
        .sgemm = library.lookup(@FieldType(Blas, "sgemm"), "cublasSgemm_v2") orelse return error.MissingBlasSymbol,
        .gemm_ex = library.lookup(@FieldType(Blas, "gemm_ex"), "cublasGemmEx") orelse return error.MissingBlasSymbol,
    };
}

pub fn close(blas: *Blas) void {
    _ = blas.destroy(blas.handle);
    blas.library.close();
}

/// Computes the row-major C[m,n] = A[m,k] B[n,k]^T in f32 from row-major f32 A and B.
pub fn reference(blas: Blas, a: cuda.Buffer(f32), b: cuda.Buffer(f32), c: cuda.Buffer(f32), m: u32, n: u32, k: u32) !void {
    const alpha: f32 = 1;
    const beta: f32 = 0;
    try check(blas.sgemm(blas.handle, operation_t, operation_n, @intCast(n), @intCast(m), @intCast(k), &alpha, b.address, @intCast(k), a.address, @intCast(k), &beta, c.address, @intCast(n)));
}

/// Computes the row-major C[m,n] = A[m,k] B[n,k]^T in f32 from row-major bf16 A and B.
pub fn gemmBf16(
    blas: Blas,
    a: cuda.Buffer(u16),
    b: cuda.Buffer(u16),
    c: cuda.Buffer(f32),
    m: u32,
    n: u32,
    k: u32,
) !void {
    const alpha: f32 = 1;
    const beta: f32 = 0;
    try check(blas.gemm_ex(
        blas.handle,
        operation_t,
        operation_n,
        @intCast(n),
        @intCast(m),
        @intCast(k),
        &alpha,
        b.address,
        data_type_bf16,
        @intCast(k),
        a.address,
        data_type_bf16,
        @intCast(k),
        &beta,
        c.address,
        data_type_f32,
        @intCast(n),
        compute_type_f32,
        algorithm_default,
    ));
}

fn check(status: c_int) !void {
    if (status != 0) {
        std.log.err("cuBLAS failed with status {d}", .{status});
        return error.BlasFailed;
    }
}
