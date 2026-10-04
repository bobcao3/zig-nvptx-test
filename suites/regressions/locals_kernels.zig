const ops = @import("locals_ops.zig");

export fn constant_table(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.constant_table(out);
}

export fn element_pointers(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.element_pointers(out);
}

export fn pointer_receiver(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.pointer_receiver(out);
}

export fn array_fields(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.array_fields(out);
}

export fn pointer_select(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    ops.pointer_select(out);
}
