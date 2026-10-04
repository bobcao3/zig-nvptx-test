//! Each kernel reads and writes `.shared` and `.local` memory through generic pointers, which the
//! release modes give their state spaces after inlining, and checks `@memset`, `@memcpy` and
//! `@memmove` in each state space. A failed check stores its line number in `out[0]`.

const builtin = @import("builtin");
const nv = @import("std").nvptx;

const Out = [*]addrspace(.global) u32;

const region_words = 64;

var regions: [128][region_words]u32 addrspace(.shared) = undefined;
var counter: u32 addrspace(.shared) = undefined;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

fn region() *[region_words]u32 {
    return @addrSpaceCast(&regions[@workItemId(0)]);
}

/// Keeps the variable in memory, because the address leaves the kernel.
noinline fn escape(comptime T: type, p: *T) void {
    const v = p.*;
    p.* = v;
}

fn fill(words: []u32, start: u32) void {
    for (words, 0..) |*word, k| word.* = start + @as(u32, @intCast(k));
}

fn sumOf(words: []const u32) u32 {
    var sum: u32 = 0;
    for (words) |word| sum +%= word;
    return sum;
}

export fn generic_shared(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const words = region();
    fill(words, t);
    expect(out, @src().line, sumOf(words) == region_words * t + region_words * (region_words - 1) / 2);
    expect(out, @src().line, words[t % region_words] == t + t % region_words);
    if (t == 0) counter = 0;
    nv.barrierSyncAligned(.work_group, null);
    const shared_counter: *u32 = @addrSpaceCast(&counter);
    _ = @atomicRmw(u32, shared_counter, .Add, t, .monotonic);
    nv.barrierSyncAligned(.work_group, null);
    expect(out, @src().line, @atomicLoad(u32, shared_counter, .acquire) == 127 * 128 / 2);
}

export fn generic_local(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    var words: [16]u32 = undefined;
    fill(&words, t);
    escape([16]u32, &words);
    const view: []u32 = words[t % 4 ..];
    expect(out, @src().line, view[t % 8] == t + t % 4 + t % 8);
    expect(out, @src().line, sumOf(&words) == 16 * t + 120);
}

const Pair = extern struct { a: u32, b: f32, c: [2]u16 };

fn copyThrough(dest: *Pair, src: *const Pair) void {
    dest.* = src.*;
}

export fn copies(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const words = region();
    const pairs: *[4]Pair = @ptrCast(words[0..16]);
    pairs[0] = .{ .a = t, .b = 1.5, .c = .{ 3, 4 } };
    var local: Pair = undefined;
    copyThrough(&local, &pairs[0]);
    escape(Pair, &local);
    copyThrough(&pairs[t % 3 + 1], &local);
    const copied = pairs[t % 3 + 1];
    expect(out, @src().line, copied.a == t and copied.b == 1.5 and copied.c[0] == 3 and copied.c[1] == 4);
    var many: [24]u32 = undefined;
    fill(&many, 2 * t);
    escape([24]u32, &many);
    const n = 17 + t % 7;
    @memcpy(words[0..n], many[0..n]);
    expect(out, @src().line, sumOf(words[0..n]) == sumOf(many[0..n]));
    expect(out, @src().line, words[n - 1] == 2 * t + n - 1);
    // The safety check of `@memcpy` compares a 32-bit `.shared` pointer with a 64-bit pointer.
    const shared_words: [*]addrspace(.shared) u32 = @ptrCast(&regions[t]);
    @memcpy(shared_words[n .. 2 * n], many[0..n]);
    expect(out, @src().line, shared_words[2 * n - 1] == 2 * t + n - 1 and shared_words[n] == 2 * t);
}

export fn memmoves(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const n = 20 + t % 5;
    var buf: [32]u32 = undefined;
    fill(&buf, 0);
    escape([32]u32, &buf);
    @memmove(buf[1 .. n + 1], buf[0..n]);
    expect(out, @src().line, buf[0] == 0 and buf[1] == 0 and buf[n] == n - 1 and buf[n + 1] == n + 1);
    @memmove(buf[0..n], buf[2 .. n + 2]);
    expect(out, @src().line, buf[0] == 1 and buf[n - 2] == n - 1 and buf[n - 1] == n + 1 and buf[n] == n - 1);
    const words = region();
    fill(words, t);
    @memmove(words[3 .. n + 3], words[0..n]);
    expect(out, @src().line, words[3] == t and words[n + 2] == t + n - 1 and words[2] == t + 2);
    var small: [6]u16 = .{ 1, 2, 3, 4, 5, 6 };
    @memmove(small[1..5], small[0..4]);
    expect(out, @src().line, small[0] == 1 and small[1] == 1 and small[4] == 4 and small[5] == 6);
    @memmove(small[0..4], small[2..6]);
    expect(out, @src().line, small[0] == 2 and small[1] == 3 and small[3] == 6);
    @memmove(words[0..n], buf[0..n]);
    expect(out, @src().line, words[0] == 1 and words[n - 1] == n + 1);
}

noinline fn regionBase() [*]u32 {
    return region();
}

export fn address_arithmetic(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const words = region();
    const p = &words[t % 16];
    p.* = 42;
    const base = regionBase();
    const offset = @intFromPtr(p) - @intFromPtr(base);
    expect(out, @src().line, offset == (t % 16) * 4);
    expect(out, @src().line, base[offset / 4] == 42);
    const bytes: [*]addrspace(.shared) u8 = @ptrCast(&regions[t]);
    const alias: [*]u8 = @addrSpaceCast(bytes);
    for (0..32) |k| bytes[k] = @intCast(k);
    const n = 20 + t % 5;
    @memmove(bytes[1 .. n + 1], alias[0..n]);
    expect(out, @src().line, bytes[0] == 0 and bytes[1] == 0 and bytes[n] == n - 1 and bytes[n + 1] == n + 1);
}

fn setThrough(comptime T: type, dest: []T, value: T) void {
    @memset(dest, value);
}

export fn memsets(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const n = 9 + t % 7;
    const words = region();
    fill(words, 0);
    setThrough(u32, words[1 .. n + 1], t);
    expect(out, @src().line, words[0] == 0 and words[1] == t and words[n] == t and words[n + 1] == n + 1);
    const bytes: *[region_words * 4]u8 = @ptrCast(words);
    setThrough(u8, bytes[3 .. n + 3], 0xab);
    expect(out, @src().line, bytes[2] == 0 and bytes[3] == 0xab and bytes[n + 2] == 0xab);
    var floats: [40]f32 = undefined;
    @memset(&floats, 2.5);
    escape([40]f32, &floats);
    setThrough(f32, floats[t % 8 ..], -1);
    expect(out, @src().line, floats[t % 8] == -1 and (t % 8 == 0 or floats[t % 8 - 1] == 2.5));
    var flags: [20]bool = undefined;
    setThrough(bool, &flags, true);
    escape([20]bool, &flags);
    setThrough(bool, flags[0..n], false);
    expect(out, @src().line, !flags[n - 1] and flags[n]);
    var wide: [6]u64 = undefined;
    setThrough(u64, &wide, 0x1234_5678_9abc_def0);
    escape([6]u64, &wide);
    expect(out, @src().line, wide[t % 6] == 0x1234_5678_9abc_def0);
}

fn addThrough(p: *u32, v: u32) u32 {
    return @atomicRmw(u32, p, .Add, v, .monotonic);
}

fn maxThrough(p: *u64, v: u64) u64 {
    return @atomicRmw(u64, p, .Max, v, .acq_rel);
}

fn minThrough(p: *i32, v: i32) i32 {
    return @atomicRmw(i32, p, .Min, v, .seq_cst);
}

fn xorThrough(p: *u64, v: u64) u64 {
    return @atomicRmw(u64, p, .Xor, v, .release);
}

fn exchangeThrough(p: *f32, v: f32) f32 {
    return @atomicRmw(f32, p, .Xchg, v, .monotonic);
}

fn addFloatThrough(p: *f64, v: f64) f64 {
    return @atomicRmw(f64, p, .Add, v, .monotonic);
}

fn addHalfThrough(p: *f16, v: f16) f16 {
    return @atomicRmw(f16, p, .Add, v, .monotonic);
}

fn nandThrough(p: *u32, v: u32) u32 {
    return @atomicRmw(u32, p, .Nand, v, .monotonic);
}

fn swapThrough(p: *u16, expected: u16, new: u16) ?u16 {
    return @cmpxchgStrong(u16, p, expected, new, .acq_rel, .acquire);
}

fn loadThrough(p: *const u32) u32 {
    return @atomicLoad(u32, p, .acquire);
}

fn storeThrough(p: *u32, v: u32) void {
    @atomicStore(u32, p, v, .release);
}

/// Debug mode does not inline, so an atomic through a generic pointer to a local variable stays an
/// `atom`, which PTX does not allow on `.local` memory. LLVM has the same limit without
/// optimization.
export fn local_atomics(out: Out) callconv(.kernel) void {
    if (builtin.mode == .debug) return;
    const t = @workItemId(0);
    var a: u32 = t;
    var b: u64 = 7;
    var c: i32 = 5;
    var d: f32 = 1.5;
    var e: f64 = 0.25;
    var h: f16 = 1.0;
    var word: u32 = 0xf0;
    var half: u16 = 9;
    escape(u32, &a);
    escape(u64, &b);
    escape(i32, &c);
    escape(f32, &d);
    escape(f64, &e);
    escape(f16, &h);
    escape(u32, &word);
    escape(u16, &half);
    expect(out, @src().line, addThrough(&a, 5) == t and a == t + 5);
    expect(out, @src().line, maxThrough(&b, t) == 7 and b == @max(7, t));
    expect(out, @src().line, minThrough(&c, -@as(i32, @intCast(t))) == 5 and c == @min(5, -@as(i32, @intCast(t))));
    expect(out, @src().line, xorThrough(&b, 3) == @max(7, t) and b == @max(7, t) ^ 3);
    expect(out, @src().line, exchangeThrough(&d, 2.5) == 1.5 and d == 2.5);
    expect(out, @src().line, addFloatThrough(&e, 0.5) == 0.25 and e == 0.75);
    expect(out, @src().line, addHalfThrough(&h, 2.0) == 1.0 and h == 3.0);
    expect(out, @src().line, nandThrough(&word, 0x3c) == 0xf0 and word == ~@as(u32, 0xf0 & 0x3c));
    expect(out, @src().line, swapThrough(&half, 9, 4) == null and half == 4);
    expect(out, @src().line, swapThrough(&half, 9, 5).? == 4 and half == 4);
    storeThrough(&a, t * 3);
    expect(out, @src().line, loadThrough(&a) == t * 3);
}
