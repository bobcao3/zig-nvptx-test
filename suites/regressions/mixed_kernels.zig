const ops = @import("mixed_ops.zig");

export fn wide_halves(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.wide_halves(out);
}

export fn narrow_stores(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.narrow_stores(out);
}

export fn sign_extension(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.sign_extension(out);
}

export fn word_bytes(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.word_bytes(out);
}

export fn float_payloads(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.float_payloads(out);
}

export fn double_halves(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.double_halves(out);
}

export fn pointer_views(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.pointer_views(out);
}

export fn packed_fields(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.packed_fields(out);
}

export fn tagged_union(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.tagged_union(out);
}

export fn conditional_stores(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.conditional_stores(out);
}

export fn unwritten_bytes(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.unwritten_bytes(out);
}
