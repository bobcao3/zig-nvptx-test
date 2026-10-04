const P = struct { a: u32, b: u16 };

var sh_f: [16]f32 addrspace(.shared) = undefined;
var sh_b: [64]u8 addrspace(.shared) = undefined;
var sh_p: [8]P addrspace(.shared) = undefined;
var sh_u: [8]u24 addrspace(.shared) = undefined;
var sh_q: [8]bool addrspace(.shared) = undefined;

fn sum8(s: []addrspace(.shared) const u8) u32 {
    var t: u32 = 0;
    for (s, 0..) |c, i| t += @as(u32, c) * @as(u32, @intCast(i + 1));
    return t;
}

export fn mem(out: [*]addrspace(.global) u32) callconv(.kernel) void {
    // memset of float elements in shared memory through an array pointer and a slice
    @memset(&sh_f, 1.5);
    @memset(sh_f[4..12], -2.0);
    var f: f32 = 0;
    for (sh_f) |x| f += x;
    out[0] = @bitCast(f);

    // memcpy global -> shared of u16 elements
    const g: [*]addrspace(.global) u16 = @ptrCast(@alignCast(out + 64));
    for (0..16) |i| g[i] = @intCast(1000 + i);
    const dst: [*]addrspace(.shared) u16 = @ptrCast(@alignCast(&sh_b));
    @memcpy(dst[0..16], g[0..16]);
    var s: u32 = 0;
    for (dst[0..16]) |x| s += x;
    out[1] = s;

    // memmove with overlap in both directions, bytes at odd offsets
    for (&sh_b, 0..) |*c, i| c.* = @intCast(i);
    @memmove(sh_b[3..43], sh_b[1..41]);
    out[2] = sum8(&sh_b);
    for (&sh_b, 0..) |*c, i| c.* = @intCast(i);
    @memmove(sh_b[1..41], sh_b[3..43]);
    out[3] = sum8(&sh_b);

    // memmove of 6-byte elements that overlap by less than an element
    for (&sh_p, 0..) |*e, i| e.* = .{ .a = @intCast(i * 10), .b = @intCast(i) };
    @memmove(sh_p[1..8], sh_p[0..7]);
    var t: u32 = 0;
    for (sh_p) |e| t = t * 3 + e.a + e.b;
    out[4] = t;

    // memset and memcpy of an integer type that is not a whole number of bytes, and bool
    @memset(&sh_u, 0xABCDE);
    var u: u32 = 0;
    for (sh_u) |x| u +%= x;
    out[5] = u;
    @memset(&sh_q, true);
    sh_q[3] = false;
    var q: u32 = 0;
    for (sh_q) |x| q = q * 2 + @intFromBool(x);
    out[6] = q;

    // global memmove of u32 and generic pointers
    for (0..32) |i| out[128 + i] = @intCast(i);
    const gen: [*]u32 = @ptrCast(@alignCast(@addrSpaceCast(out + 128)));
    @memmove(gen[2..22], gen[0..20]);
    out[7] = out[128 + 2] * 1000 + out[128 + 21];
}
