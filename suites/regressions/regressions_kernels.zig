const ops = @import("regressions_ops.zig");

export fn zero_extend(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.zero_extend(out);
}

export fn float_div_trunc(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.float_div_trunc(out);
}

export fn vector_u4(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vector_u4(out);
}

export fn vector_u14(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vector_u14(out);
}

export fn vector_memory(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.vector_memory(out);
}

export fn counted_loops(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.counted_loops(out);
}

export fn large_arguments(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.large_arguments(out);
}

export fn local_pointer_choice(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.local_pointer_choice(out);
}

export fn aggregate_loads(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.aggregate_loads(out);
}

export fn value_choices(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.value_choices(out);
}
