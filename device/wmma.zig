//! This file implements the `wmma` functions of the CUDA sample ports on top of
//! `std.nvptx.mmaSync` and `std.nvptx.ldmatrix`, with the names of the PTX `wmma` instructions. A
//! fragment holds the registers of `mma.sync` in the layouts of PTX ISA 9.4, section 9.7.16.5. A
//! 16 by 16 tile is two 16 by 8 products, and an 8 by 8 tile is one.

const std = @import("std");
const nvptx = std.nvptx;
const matrix = nvptx.matrix;

pub const Shape = matrix.Shape;
pub const Layout = matrix.Layout;
pub const Element = matrix.Element;
pub const Accumulator = matrix.Accumulator;

pub const Operand = union(enum) {
    a: Element,
    b: Element,
    c: Accumulator,
};

pub const Options = struct {
    element: Element,
    accumulator: Accumulator,
};

/// A sample with many inline calls in one function exceeds the default quota.
const eval_branch_quota = 100_000;

const Product = struct {
    shape: matrix.MmaSyncShape,
    halves: comptime_int,
};

fn product(comptime shape: Shape, comptime element: Element) Product {
    const used: Product = switch (element) {
        .f16, .bf16, .s8, .u8 => .{ .shape = .m16n8k16, .halves = 2 },
        .tf32 => .{ .shape = .m16n8k8, .halves = 2 },
        .f64 => .{ .shape = .m8n8k4, .halves = 1 },
        else => unsupported(shape, element),
    };
    const size = used.shape.dimensions();
    if (shape.m != size.m or shape.n != size.n * used.halves or shape.k != size.k) {
        unsupported(shape, element);
    }
    return used;
}

fn unsupported(comptime shape: Shape, comptime element: Element) noreturn {
    @compileError(std.fmt.comptimePrint(
        "wmma has no shape m{d}n{d}k{d} for {t}",
        .{ shape.m, shape.n, shape.k, element },
    ));
}

fn checkElements(comptime Pointer: type, comptime bit_count: u16) void {
    if (@bitSizeOf(@typeInfo(Pointer).pointer.child) != bit_count) {
        @compileError(std.fmt.comptimePrint("expected {d}-bit elements, found '{s}'", .{
            bit_count,
            @typeName(Pointer),
        }));
    }
}

fn bits(comptime element: Element) u16 {
    return switch (element) {
        .f16, .bf16 => 16,
        .s8, .u8 => 8,
        .tf32 => 32,
        .f64 => 64,
        else => unreachable,
    };
}

fn Registers(comptime element: Element, comptime elements: u32) type {
    if (element == .f64) return @Vector(elements, f64);
    return @Vector(elements * bits(element) / 32, u32);
}

fn Scalar(comptime accumulator: Accumulator) type {
    return switch (accumulator) {
        .f32 => f32,
        .s32 => i32,
        .f64 => f64,
        .f16 => @compileError("wmma has no f16 accumulator"),
    };
}

pub fn Fragment(comptime shape: Shape, comptime operand: Operand) type {
    return switch (operand) {
        .a => |element| Registers(element, shape.m * shape.k / 32),
        .b => |element| Registers(element, shape.k * shape.n / 32),
        .c => |accumulator| @Vector(shape.m * shape.n / 32, Scalar(accumulator)),
    };
}

fn space(comptime Pointer: type) std.lang.AddressSpace {
    return @typeInfo(Pointer).pointer.attrs.@"addrspace" orelse .generic;
}

inline fn read(comptime T: type, pointer: anytype, index: u32) T {
    const Typed = *addrspace(space(@TypeOf(pointer))) const T;
    const typed: Typed = @ptrCast(@alignCast(&pointer[index]));
    return typed.*;
}

inline fn write(comptime T: type, pointer: anytype, index: u32, value: T) void {
    const Typed = *addrspace(space(@TypeOf(pointer))) T;
    const typed: Typed = @ptrCast(@alignCast(&pointer[index]));
    typed.* = value;
}

fn offset(layout: Layout, row: u32, column: u32, stride: u32) u32 {
    return switch (layout) {
        .row => row * stride + column,
        .col => column * stride + row,
    };
}

/// Returns the row of the accumulator element `i` of a work item, as `mma.sync` places it.
fn accumulatorRow(lane: u32, i: u32) u32 {
    return lane / 4 + 8 * (i % 4 / 2);
}

/// Returns the column of the accumulator element `i` of a work item, as `mma.sync` places it.
fn accumulatorColumn(lane: u32, i: u32) u32 {
    return i / 4 * 8 + lane % 4 * 2 + i % 2;
}

/// Loads the `operand` fragment of a matrix at `pointer` whose rows (`.row`) or columns (`.col`)
/// lie `stride` elements apart. A 16-bit load from shared memory uses `ldmatrix`, which assumes
/// 16-byte aligned rows.
pub inline fn loadSyncAligned(
    comptime shape: Shape,
    comptime operand: Operand,
    comptime layout: Layout,
    pointer: anytype,
    stride: u32,
) Fragment(shape, operand) {
    @setEvalBranchQuota(eval_branch_quota);
    const lane = nvptx.laneid();
    var fragment: Fragment(shape, operand) = undefined;
    switch (operand) {
        .c => |accumulator| {
            checkElements(@TypeOf(pointer), @bitSizeOf(Scalar(accumulator)));
            inline for (0..@typeInfo(@TypeOf(fragment)).vector.len) |i| {
                const row = accumulatorRow(lane, i);
                const index = offset(layout, row, accumulatorColumn(lane, i), stride);
                fragment[i] = read(Scalar(accumulator), pointer, index);
            }
        },
        .a, .b => |element| {
            const is_a = operand == .a;
            const used = comptime product(shape, element);
            checkElements(@TypeOf(pointer), bits(element));
            const contiguous = layout == if (is_a) Layout.row else Layout.col;
            const shared = comptime space(@TypeOf(pointer)) == .shared;
            if (comptime shared and bits(element) == 16 and contiguous) {
                const row = if (is_a) lane % 16 else lane % 8 + lane / 16 * 8;
                const column = if (is_a) lane / 16 * 8 else lane / 8 % 2 * 8;
                return nvptx.ldmatrix(.x4, false, &pointer[row * stride + column]);
            }
            const per = if (element == .f64) 1 else 32 / bits(element);
            const Bits = @Int(.unsigned, bits(element));
            const half_registers = @typeInfo(@TypeOf(fragment)).vector.len / used.halves;
            const rows_per_half = shape.m / 8;
            inline for (0..@typeInfo(@TypeOf(fragment)).vector.len) |r| {
                const half: u32 = r / half_registers;
                const register: u32 = if (is_a) r else r % half_registers;
                const block = if (is_a) register / rows_per_half else register;
                const k = block * 4 * per + lane % 4 * per;
                const other = if (is_a)
                    lane / 4 + 8 * (register % rows_per_half)
                else
                    half * 8 + lane / 4;
                if (element == .f64 or contiguous) {
                    const index = if (is_a)
                        offset(layout, other, k, stride)
                    else
                        offset(layout, k, other, stride);
                    fragment[r] = read(@typeInfo(@TypeOf(fragment)).vector.child, pointer, index);
                } else {
                    var word: u32 = 0;
                    inline for (0..per) |element_index| {
                        const e: u32 = element_index;
                        const index = if (is_a)
                            offset(layout, other, k + e, stride)
                        else
                            offset(layout, k + e, other, stride);
                        word |= @as(u32, read(Bits, pointer, index)) << @intCast(e * bits(element));
                    }
                    fragment[r] = word;
                }
            }
        },
    }
    return fragment;
}

/// Returns D = A * B + C for one tile.
pub inline fn mmaSyncAligned(
    comptime shape: Shape,
    comptime options: Options,
    a: Fragment(shape, .{ .a = options.element }),
    b: Fragment(shape, .{ .b = options.element }),
    c: Fragment(shape, .{ .c = options.accumulator }),
) @TypeOf(c) {
    @setEvalBranchQuota(eval_branch_quota);
    const used = comptime product(shape, options.element);
    const mma_opt: matrix.MmaOptions = .{
        .shape = used.shape,
        .accumulator = options.accumulator,
        .element = options.element,
    };
    return if (used.halves == 1) nvptx.mmaSync(mma_opt, a, b, c) else mmaTwice(mma_opt, a, b, c);
}

inline fn mmaTwice(
    comptime mma_opt: matrix.MmaOptions,
    a: anytype,
    b: anytype,
    c: anytype,
) @TypeOf(c) {
    const b_len = @typeInfo(@TypeOf(b)).vector.len / 2;
    const c_len = @typeInfo(@TypeOf(c)).vector.len / 2;
    const B = @Vector(b_len, @typeInfo(@TypeOf(b)).vector.child);
    const T = @typeInfo(@TypeOf(c)).vector.child;
    const b_array: [2 * b_len]@typeInfo(@TypeOf(b)).vector.child = b;
    const c_array: [2 * c_len]T = c;
    const C = @Vector(c_len, T);
    const b_low: B = b_array[0..b_len].*;
    const b_high: B = b_array[b_len..].*;
    const c_low: C = c_array[0..c_len].*;
    const c_high: C = c_array[c_len..].*;
    const low: [c_len]T = nvptx.mmaSync(mma_opt, a, b_low, c_low);
    const high: [c_len]T = nvptx.mmaSync(mma_opt, a, b_high, c_high);
    return low ++ high;
}

/// Stores `d` to the matrix at `pointer` and `stride`, which mean the same as in `loadSyncAligned`.
pub inline fn storeSyncAligned(
    comptime shape: Shape,
    comptime accumulator: Accumulator,
    comptime layout: Layout,
    pointer: anytype,
    d: Fragment(shape, .{ .c = accumulator }),
    stride: u32,
) void {
    @setEvalBranchQuota(eval_branch_quota);
    checkElements(@TypeOf(pointer), @bitSizeOf(Scalar(accumulator)));
    const lane = nvptx.laneid();
    inline for (0..@typeInfo(@TypeOf(d)).vector.len) |i| {
        const index = offset(layout, accumulatorRow(lane, i), accumulatorColumn(lane, i), stride);
        write(Scalar(accumulator), pointer, index, d[i]);
    }
}
