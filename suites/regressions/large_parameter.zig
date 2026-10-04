const Parameter = extern struct { a: [2000]i32 };

export fn sum(p: Parameter, out: *addrspace(.global) i32) callconv(.kernel) void {
    var total: i32 = 0;
    for (p.a) |x| total +%= x;
    out.* = total;
}
