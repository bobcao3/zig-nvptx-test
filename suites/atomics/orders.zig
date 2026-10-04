const G = *addrspace(.global) u32;
const S = *addrspace(.shared) u32;
var shared_word: u32 addrspace(.shared) = undefined;

export fn loads(p: G, out: G) callconv(.kernel) void {
    out.* = @atomicLoad(u32, p, .unordered);
    out.* += @atomicLoad(u32, p, .monotonic);
    out.* += @atomicLoad(u32, p, .acquire);
    out.* += @atomicLoad(u32, p, .seq_cst);
}

export fn stores(p: G) callconv(.kernel) void {
    @atomicStore(u32, p, 1, .unordered);
    @atomicStore(u32, p, 2, .monotonic);
    @atomicStore(u32, p, 3, .release);
    @atomicStore(u32, p, 4, .seq_cst);
}

export fn rmws(p: G, out: G) callconv(.kernel) void {
    out.* = @atomicRmw(u32, p, .Add, 1, .monotonic);
    out.* += @atomicRmw(u32, p, .Add, 1, .acquire);
    out.* += @atomicRmw(u32, p, .Add, 1, .release);
    out.* += @atomicRmw(u32, p, .Add, 1, .acq_rel);
    out.* += @atomicRmw(u32, p, .Add, 1, .seq_cst);
}

export fn cas(p: G, out: G) callconv(.kernel) void {
    out.* = if (@cmpxchgStrong(u32, p, 1, 2, .monotonic, .monotonic)) |_| 1 else 0;
    out.* += if (@cmpxchgStrong(u32, p, 1, 2, .acq_rel, .acquire)) |_| 1 else 0;
    out.* += if (@cmpxchgStrong(u32, p, 1, 2, .release, .acquire)) |_| 1 else 0;
    out.* += if (@cmpxchgWeak(u32, p, 1, 2, .seq_cst, .seq_cst)) |_| 1 else 0;
}

export fn ops(p: G, out: G) callconv(.kernel) void {
    out.* = @atomicRmw(u32, p, .Xchg, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .Sub, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .And, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .Nand, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .Or, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .Xor, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .Max, 3, .monotonic);
    out.* += @atomicRmw(u32, p, .Min, 3, .monotonic);
}

export fn sharedOps(out: G) callconv(.kernel) void {
    out.* = @atomicRmw(u32, &shared_word, .Add, 1, .acq_rel);
    out.* += @atomicLoad(u32, &shared_word, .acquire);
    @atomicStore(u32, &shared_word, 5, .release);
}

export fn small(p: *addrspace(.global) u8, q: *addrspace(.global) u16, out: G) callconv(.kernel) void {
    out.* = @atomicRmw(u8, p, .Add, 1, .acq_rel);
    out.* += @atomicRmw(u16, q, .Max, 1, .monotonic);
}

export fn floats(f: *addrspace(.global) f32, g: *addrspace(.global) f64, h: *addrspace(.global) f16) callconv(.kernel) void {
    _ = @atomicRmw(f32, f, .Add, 1, .monotonic);
    _ = @atomicRmw(f64, g, .Add, 1, .monotonic);
    _ = @atomicRmw(f16, h, .Add, 1, .monotonic);
    _ = @atomicRmw(f32, f, .Max, 1, .monotonic);
}
