const ops = @import("int_ops.zig");

export fn add(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.add(out);
}

export fn sub(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.sub(out);
}

export fn mul(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.mul(out);
}

export fn shl(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.shl(out);
}

export fn add_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.add_sat(out);
}

export fn sub_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.sub_sat(out);
}

export fn mul_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.mul_sat(out);
}

export fn shl_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.shl_sat(out);
}

export fn shl_sat_wide(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.shl_sat_wide(out);
}

export fn clz(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.clz(out);
}

export fn ctz(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.ctz(out);
}

export fn popcount(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.popcount(out);
}

export fn bit_reverse(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.bit_reverse(out);
}

export fn byte_swap(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.byte_swap(out);
}

export fn abs(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.abs(out);
}

export fn vec_add(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_add(out);
}

export fn vec_sub(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_sub(out);
}

export fn vec_mul(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_mul(out);
}

export fn vec_shl(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_shl(out);
}

export fn vec_add_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_add_sat(out);
}

export fn vec_sub_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_sub_sat(out);
}

export fn vec_mul_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_mul_sat(out);
}

export fn vec_shl_sat(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_shl_sat(out);
}

export fn vec_clz(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_clz(out);
}

export fn vec_ctz(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_ctz(out);
}

export fn vec_popcount(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_popcount(out);
}

export fn vec_bit_reverse(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_bit_reverse(out);
}

export fn vec_byte_swap(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_byte_swap(out);
}

export fn vec_abs(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_abs(out);
}
