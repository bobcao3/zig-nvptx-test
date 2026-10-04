//! Each kernel reads elements of vectors whose elements have fewer bits than their bytes hold, so
//! the vectors have a bit-packed layout. A failed check stores its line number in `out[0]`.

const Out = [*]addrspace(.global) u32;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

noinline fn hide(comptime T: type, x: T) T {
    return x;
}

noinline fn make(a: u12) @Vector(8, u12) {
    return .{ a, 1, 2, 3, 4, 0x5d6, 6, 7 };
}

export fn indexed_literals(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const a = hide(u12, @intCast(t));
    expect(out, @src().line, (@Vector(8, u12){ a, 1, 2, 3, 4, 0x5d6, 6, 7 })[5] == 0x5d6);
    expect(out, @src().line, (@Vector(8, u12){ 9, 1, 2, a, 4, 5, 6, 7 })[3] == a);
    const b = hide(u63, t);
    expect(out, @src().line, (@Vector(4, u63){ b, 0, 1, 0x7fff_ffff_ffff_ffff })[3] == 0x7fff_ffff_ffff_ffff);
    expect(out, @src().line, (@Vector(4, u63){ 5, 0x1_ffff_ffff, b, 1 })[2] == b);
    expect(out, @src().line, (@Vector(5, bool){ true, false, a == 0, true, false })[2] == (a == 0));
    expect(out, @src().line, (@Vector(3, i7){ -64, @intCast(t % 64), 63 })[1] == t % 64);
    const c = hide(u100, t);
    expect(out, @src().line, (@Vector(3, u100){ 1 << 99, c, 3 })[1] == c);
    expect(out, @src().line, (@Vector(3, u100){ 1 << 99, c, 3 })[0] == 1 << 99);
}

export fn initialized_variables(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const a = hide(u12, @intCast(t));
    var v: @Vector(8, u12) = .{ a, 1, 2, 3, 4, 0x5d6, 6, 7 };
    v[2] = a +% 1;
    expect(out, @src().line, v[5] == 0x5d6 and v[2] == a +% 1 and v[0] == a);
    const made = make(a);
    expect(out, @src().line, made[5] == 0x5d6 and made[7] == 7 and made[0] == a);
    const sum = @reduce(.Add, made);
    expect(out, @src().line, sum == a +% 1 +% 2 +% 3 +% 4 +% 0x5d6 +% 6 +% 7);
    var w: @Vector(4, u63) = .{ hide(u63, t), 0, 0x1_ffff_ffff, 0x1_ffff_ffff };
    w[1] = w[3] + 1;
    expect(out, @src().line, w[1] == 0x2_0000_0000 and w[3] == 0x1_ffff_ffff and w[0] == t);
}
