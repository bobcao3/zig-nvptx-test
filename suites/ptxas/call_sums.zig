//! Each kernel adds 64-bit constants outside the signed 32-bit range to the results of two calls.
//! A failed check stores its line number in `out[0]`.

const Out = [*]addrspace(.global) u32;

const threads = 128;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

noinline fn hide(x: u64) u64 {
    return x;
}

noinline fn fmix64(x0: u64) u64 {
    var x = x0;
    x ^= x >> 33;
    x *%= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    x *%= 0xc4ceb9fe1a85ec53;
    return x ^ (x >> 33);
}

noinline fn tripled(x: u128) u128 {
    return x *% 3;
}

fn offsetKeys(t: u64) [4]u64 {
    const base: u64 = 1 << 40;
    const range = hide(1_000_000_007 + t * 0x1_0000_0001);
    return .{
        base +% fmix64(2 * t),
        base +% fmix64(2 * t + 1),
        0x0fff_ffff_ffff_ffff +% fmix64(3 * t) % range,
        0x0fff_ffff_ffff_ffff +% fmix64(3 * t + 1) % range,
    };
}

export fn offset_keys(out: Out) callconv(.kernel) void {
    @setEvalBranchQuota(100_000);
    const expected = comptime table: {
        var keys: [threads][4]u64 = undefined;
        for (&keys, 0..) |*row, t| row.* = offsetKeys(t);
        break :table keys;
    };
    const t = @workItemId(0);
    const keys = offsetKeys(t);
    for (keys, expected[t]) |key, want| expect(out, @src().line, key == want);
    const wide = tripled(@as(u128, hide(t)) << 64 | hide(fmix64(t))) +% (1 << 40);
    expect(out, @src().line, wide == comptime_wide[t]);
}

const comptime_wide = table: {
    @setEvalBranchQuota(100_000);
    var sums: [threads]u128 = undefined;
    for (&sums, 0..) |*sum, t| sum.* = tripled(@as(u128, t) << 64 | fmix64(t)) +% (1 << 40);
    break :table sums;
};
