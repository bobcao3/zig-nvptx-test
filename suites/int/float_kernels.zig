const ops = @import("float_ops.zig");

export fn round_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.round_f16(out);
}

export fn round_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.round_f32(out);
}

export fn round_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.round_f64(out);
}

export fn rem_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.rem_f16(out);
}

export fn rem_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.rem_f32(out);
}

export fn rem_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.rem_f64(out);
}

export fn mod_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.mod_f16(out);
}

export fn mod_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.mod_f32(out);
}

export fn mod_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.mod_f64(out);
}

export fn neg_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.neg_f16(out);
}

export fn neg_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.neg_f32(out);
}

export fn neg_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.neg_f64(out);
}

export fn abs_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.abs_f16(out);
}

export fn abs_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.abs_f32(out);
}

export fn abs_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.abs_f64(out);
}

export fn vec_round_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_round_f16(out);
}

export fn vec_round_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_round_f32(out);
}

export fn vec_round_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_round_f64(out);
}

export fn vec_rem_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_rem_f16(out);
}

export fn vec_rem_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_rem_f32(out);
}

export fn vec_rem_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_rem_f64(out);
}

export fn vec_mod_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_mod_f16(out);
}

export fn vec_mod_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_mod_f32(out);
}

export fn vec_mod_f64(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_mod_f64(out);
}

export fn vec_neg_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_neg_f16(out);
}

export fn vec_neg_f32(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_neg_f32(out);
}

export fn vec_abs_f16(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vec_abs_f16(out);
}
