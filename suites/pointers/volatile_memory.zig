//! Each kernel fills, copies and moves memory through volatile pointers and then reads the result
//! through plain pointers. A failed check stores its line number in `out[0]`.

const nv = @import("std").nvptx;

const Out = [*]addrspace(.global) u32;
const Words = [*]addrspace(.shared) volatile u32;

var words: [2048]u32 addrspace(.shared) = undefined;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

fn slot(t: u32) Words {
    return @ptrCast(&words[8 * (t % 128)]);
}

export fn fill_array(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const dest: *addrspace(.shared) volatile [4]u32 = @ptrCast(slot(t));
    @memset(dest, 1);
    @memset(dest, 2);
    for (0..4) |i| expect(out, @src().line, dest[i] == 2);
}

export fn fill_large_array(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const dest: *addrspace(.shared) volatile [2048]u32 = @ptrCast(&words);
    if (t == 0) @memset(dest, 7);
    nv.barrierSyncAligned(.work_group, null);
    expect(out, @src().line, words[(t * 13) % 2048] == 7);
    expect(out, @src().line, words[2047] == 7);
}

export fn fill_slice(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const n = 1 + t % 8;
    const dest = slot(t);
    @memset(dest[0..8], 9);
    @memset(dest[0..n], 3);
    for (0..8) |i| expect(out, @src().line, dest[i] == @as(u32, if (i < n) 3 else 9));
}

export fn copy_array(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const source: *addrspace(.shared) volatile [4]u32 = @ptrCast(slot(t));
    const dest: *addrspace(.shared) volatile [4]u32 = @ptrCast(slot(t) + 4);
    for (0..4) |i| source[i] = t + @as(u32, @intCast(i));
    @memcpy(dest, source);
    for (0..4) |i| expect(out, @src().line, dest[i] == t + i);
}

export fn copy_slice(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const n = 1 + t % 4;
    const source = slot(t);
    const dest = slot(t) + 4;
    for (0..8) |i| source[i] = t * 8 + @as(u32, @intCast(i));
    @memcpy(dest[0..n], source[0..n]);
    for (0..n) |i| expect(out, @src().line, dest[i] == t * 8 + i);
}

export fn move_forward_and_backward(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const base = slot(t);
    for (0..8) |i| base[i] = @as(u32, @intCast(i));
    @memmove(base[2..8], base[0..6]);
    for (2..8) |i| expect(out, @src().line, base[i] == i - 2);
    for (0..8) |i| base[i] = @as(u32, @intCast(i));
    @memmove(base[0..6], base[2..8]);
    for (0..6) |i| expect(out, @src().line, base[i] == i + 2);
}

export fn volatile_source_plain_destination(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const source: *addrspace(.shared) volatile [4]u32 = @ptrCast(slot(t));
    const dest: *addrspace(.shared) [4]u32 = @ptrCast(&words[8 * (t % 128) + 4]);
    for (0..4) |i| source[i] = t * 4 + @as(u32, @intCast(i));
    @memcpy(dest, source);
    for (0..4) |i| expect(out, @src().line, dest[i] == t * 4 + i);
}

export fn plain_source_volatile_destination(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const source: *addrspace(.shared) [4]u32 = @ptrCast(&words[8 * (t % 128)]);
    const dest: *addrspace(.shared) volatile [4]u32 = @ptrCast(slot(t) + 4);
    for (0..4) |i| source[i] = t * 5 + @as(u32, @intCast(i));
    @memcpy(dest, source);
    for (0..4) |i| expect(out, @src().line, dest[i] == t * 5 + i);
}
