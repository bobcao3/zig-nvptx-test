const h = @import("harness.zig");

const Pair = struct { a: u32, b: u64 };
const Fn = *const fn (u32) u32;

fn double(x: u32) u32 {
    return x *% 2;
}

fn triple(x: u32) u32 {
    return x *% 3;
}

noinline fn square(x: u32) u32 {
    return x *% x;
}

noinline fn addPair(p: Pair) u64 {
    return p.a +% p.b;
}

noinline fn swap(p: Pair) Pair {
    return .{ .a = @truncate(p.b), .b = p.a };
}

noinline fn sumSlice(s: []const u32) u32 {
    var total: u32 = 0;
    for (s) |v| total +%= v;
    return total;
}

var counter: u32 = 0;

noinline fn tick() void {
    counter +%= 1;
}

noinline fn apply(f: Fn, x: u32) u32 {
    return f(x) +% 1;
}

noinline fn applyTwice(f: *const fn (u32) u32, x: u32) u32 {
    return f(f(x));
}

noinline fn applyPair(f: *const fn (Pair) u64, p: Pair) u64 {
    return f(p);
}

noinline fn applySwap(f: *const fn (Pair) Pair, p: Pair) Pair {
    return f(p);
}

noinline fn applySlice(f: *const fn ([]const u32) u32, s: []const u32) u32 {
    return f(s);
}

const Ops = struct {
    name: u32,
    run: Fn,
    maybe: ?Fn,
};

const ops = [_]Ops{
    .{ .name = 1, .run = &double, .maybe = &triple },
    .{ .name = 2, .run = &triple, .maybe = null },
    .{ .name = 3, .run = &square, .maybe = &double },
};

const table = [_]Fn{ &double, &triple, &square };
var var_table = [_]Fn{ &square, &double, &triple };
var stored: Fn = &triple;
var maybe_stored: ?*const fn () void = null;
const words = [_]u32{ 1, 2, 3, 4 };

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    for (0..3) |i| {
        out[k] = table[(i +% t) % 3](10);
        k +%= 1;
    }
    for (0..3) |i| {
        out[k] = var_table[(i +% t) % 3](7);
        k +%= 1;
    }
    out[k] = apply(table[t], 5);
    k +%= 1;
    out[k] = apply(&double, 5);
    k +%= 1;
    out[k] = applyTwice(table[1 +% t], 5);
    k +%= 1;
    out[k] = stored(4);
    k +%= 1;
    stored = &square;
    out[k] = stored(4);
    k +%= 1;
    stored = var_table[2 +% t];
    out[k] = stored(4);
    k +%= 1;
    out[k] = @intFromBool(stored == &triple);
    k +%= 1;
    out[k] = @intFromBool(stored == &square);
    k +%= 1;

    for (ops) |op| {
        out[k] = op.run(10 +% t) *% op.name;
        k +%= 1;
        out[k] = if (op.maybe) |m| m(10) else 0xffff;
        k +%= 1;
    }

    const pair: Pair = .{ .a = 5 +% t, .b = 0x100000000 +% @as(u64, t) };
    out[k] = @truncate(applyPair(&addPair, pair));
    k +%= 1;
    const swapped = applySwap(&swap, pair);
    out[k] = swapped.a;
    k +%= 1;
    out[k] = @truncate(swapped.b);
    k +%= 1;
    out[k] = applySlice(&sumSlice, words[t..]);
    k +%= 1;

    const direct: Fn = &double;
    out[k] = direct(21 +% t);
    k +%= 1;
    var voidfn: *const fn () void = &tick;
    voidfn();
    voidfn();
    maybe_stored = voidfn;
    if (maybe_stored) |f| f();
    voidfn = &tick;
    out[k] = counter;
    k +%= 1;
    out[k] = @intFromBool(maybe_stored != null);
    k +%= 1;
}

pub const H = h.Harness(run, 29);
comptime {
    _ = H;
}
