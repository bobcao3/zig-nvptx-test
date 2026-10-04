const ops = @import("loops_ops.zig");

export fn float_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.float_f16(out);
}

export fn float_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.float_f32(out);
}

export fn float_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.float_f64(out);
}

export fn int_i8(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.int_i8(out);
}

export fn int_u32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.int_u32(out);
}

export fn int_i32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.int_i32(out);
}

export fn int_u64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.int_u64(out);
}

export fn array_u32_8(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_u32_8(out);
}

export fn array_u32_32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_u32_32(out);
}

export fn array_u32_33(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_u32_33(out);
}

export fn array_u8_32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_u8_32(out);
}

export fn array_u8_33(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_u8_33(out);
}

export fn array_f64_16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_f64_16(out);
}

export fn array_f64_17(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_f64_17(out);
}

export fn array_pair_16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_pair_16(out);
}

export fn array_pair_17(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_pair_17(out);
}

export fn array_sentinel(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_sentinel(out);
}
