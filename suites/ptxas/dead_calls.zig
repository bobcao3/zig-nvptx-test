//! Each kernel makes a call that never runs, from a recursive function or from a function whose
//! address the module takes. A failed check stores its line number in `out[0]`.

const Out = [*]addrspace(.global) u32;

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

var reports: u32 addrspace(.global) = 0;

noinline fn report(n: u32) void {
    reports +%= n;
}

fn countDown(n: u32, x: u16) u32 {
    if (n == 0) return 0;
    const high: u64 = @truncate(@as(u128, x) >> 100);
    if (high == 3) report(n);
    return 1 + countDown(n - 1, x);
}

export fn recursion_with_dead_call(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    expect(out, @src().line, countDown(t % 8, @truncate(t)) == t % 8);
}

fn scaled(x: u32) u32 {
    const high: u64 = @truncate(@as(u128, x) >> 100);
    if (high == 3) report(x);
    return x *% 2;
}

fn negated(x: u32) u32 {
    return 0 -% x;
}

const dispatch = [_]*const fn (u32) u32{ scaled, negated };

export fn function_table_with_dead_call(out: Out) callconv(.kernel) void {
    const t = @workItemId(0);
    const result = dispatch[t & 1](21);
    expect(out, @src().line, result == if (t & 1 == 0) 42 else 0 -% @as(u32, 21));
}
