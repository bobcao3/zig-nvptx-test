//! Each function checks one language rule that the PTX backend once got wrong. The functions run
//! unchanged on the host and on the GPU and store their results in `out`. The operands come from
//! a runtime loop, so the compiler cannot fold the operations.

const samples = [_]u64{
    0,          1,                  0x7fffffff,         0x80000000,
    0xffffffff, 0x123456789abcdef0, 0xfedcba9876543210, 0xffffffffffffffff,
};

pub fn zero_extend(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const v: u32 = @truncate(sample);
        const wide: i38 = v;
        const narrow: i33 = v;
        out[n] = @intFromBool(wide >= 0);
        out[n + 1] = @truncate(@as(u38, @bitCast(wide)) >> 32);
        out[n + 2] = @intFromBool(narrow >= 0);
        out[n + 3] = @truncate(@as(u33, @bitCast(narrow)) >> 32);
        n += 4;
    }
}

const quotients = [_]f64{ 255.0, -255.0, 127.5, -127.5, 7.0, -7.0, 0.5, -0.5, 1e10, -1e10 };

/// `@divTrunc` on floats rounds the quotient toward zero.
pub fn float_div_trunc(out: anytype) void {
    var n: usize = 0;
    for (quotients) |dividend| {
        const single: f32 = @floatCast(dividend);
        const double: f64 = dividend;
        const double_bits: u64 = @bitCast(@divTrunc(double, 2.0));
        out[n] = @bitCast(@divTrunc(single, 2.0));
        out[n + 1] = @truncate(double_bits);
        out[n + 2] = @truncate(double_bits >> 32);
        n += 3;
    }
}

inline fn hashElements(vector: anytype) u32 {
    var hash: u32 = 0;
    inline for (0..@typeInfo(@TypeOf(vector)).vector.len) |i| {
        hash = hash *% 31 +% vector[i];
    }
    return hash;
}

pub fn vector_u4(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x: u32 = @truncate(sample);
        const to_vector: @Vector(8, u4) = @bitCast(x);
        var elements: [8]u4 = undefined;
        inline for (&elements, 0..) |*element, i| element.* = @truncate(x >> (3 * i));
        const from_vector: u32 = @bitCast(@as(@Vector(8, u4), elements));
        const last: @Vector(8, u4) = .{ 0, 0, 0, 0, 0, 0, 0, @truncate(x) };
        out[n] = hashElements(to_vector);
        out[n + 1] = from_vector;
        out[n + 2] = @as(u32, @bitCast(last));
        n += 3;
    }
}

pub fn vector_u14(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const x: i56 = @truncate(@as(i64, @bitCast(sample)));
        const to_vector: @Vector(4, u14) = @bitCast(x);
        var elements: [4]u14 = undefined;
        inline for (&elements, 0..) |*element, i| element.* = @truncate(sample >> (5 * i));
        const from_vector: i56 = @bitCast(@as(@Vector(4, u14), elements));
        const splat: i56 = @bitCast(@as(@Vector(4, u14), @splat(@truncate(sample))));
        const top: i12 = @intCast(from_vector >> 44);
        const from_bits: u56 = @bitCast(from_vector);
        out[n] = hashElements(to_vector);
        out[n + 1] = @truncate(from_bits);
        out[n + 2] = @truncate(from_bits >> 32);
        out[n + 3] = @as(u12, @bitCast(top));
        out[n + 4] = @truncate(@as(u56, @bitCast(splat)) >> 24);
        n += 5;
    }
}

fn At(comptime Element: type, comptime T: type, comptime alignment: usize) type {
    var attrs = @typeInfo(Element).pointer.attrs;
    attrs.@"align" = alignment;
    return @Pointer(.one, attrs, T, null);
}

fn at(words: anytype, index: usize, comptime T: type, comptime alignment: usize) At(@TypeOf(&words[0]), T, alignment) {
    return @ptrCast(@alignCast(&words[index]));
}

/// The PTX backend may only merge loads and stores into one vector access when the pointer
/// alignment proves that the address is a multiple of the access size.
pub fn vector_memory(out: anytype) void {
    const scratch = out[512..];
    const words = scratch[(16 - @intFromPtr(scratch) % 16) % 16 / 4 ..];
    for (0..64) |i| words[i] = @as(u32, @intCast(i)) *% 0x9e3779b9 +% 1;
    const Quad = extern struct { a: u32, b: u32, c: u32, d: u32 };
    at(words, 32, @Vector(4, u32), 16).* = at(words, 0, @Vector(4, u32), 16).*;
    at(words, 37, @Vector(4, u32), 4).* = at(words, 1, @Vector(4, u32), 4).*;
    at(words, 44, Quad, 16).* = at(words, 8, Quad, 16).*;
    at(words, 50, [4]u32, 4).* = at(words, 14, [4]u32, 4).*;
    at(words, 56, @Vector(2, u64), 16).* = at(words, 20, @Vector(2, u64), 16).*;
    at(words, 60, @Vector(2, u32), 8).* = at(words, 6, @Vector(2, u32), 8).*;
    for (0..64) |i| out[i] = words[i];
}

/// For each sample, writes four words from the loops and the upper half of a 64-bit sum.
pub fn counted_loops(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const bound: u32 = @truncate(sample % 37);
        const start: u32 = @truncate(sample % 5);
        var sum: u32 = 0;
        var last: u32 = 0;
        for (0..bound) |i| {
            const j: u32 = @intCast(i);
            sum +%= j *% 0x9e3779b9;
            last = j;
        }
        var count: u32 = 0;
        var i: usize = start;
        while (i < bound) : (i += 1) count += 1;
        var high: usize = 0;
        for (0..16) |k| high +%= k << 30;
        out[n] = sum;
        out[n + 1] = last;
        out[n + 2] = count;
        out[n + 3] = @truncate(i);
        out[n + 4] = @truncate(high >> 32);
        n += 5;
    }
}

const Wide = struct { words: [48]u32, tag: u64 };

noinline fn readWide(wide: Wide, index: usize) u32 {
    return wide.words[index % 48] +% @as(u32, @truncate(wide.tag));
}

noinline fn sumWide(wide: Wide) u32 {
    var sum: u32 = 0;
    for (wide.words) |word| sum +%= word;
    return sum;
}

noinline fn passOn(wide: Wide, index: usize) u32 {
    return readWide(wide, index +% 1) ^ sumWide(wide);
}

/// Passes a struct of 200 bytes by value to functions that are not inline. For each sample,
/// writes three words.
pub fn large_arguments(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        var wide: Wide = undefined;
        for (&wide.words, 0..) |*word, i| word.* = @truncate(sample *% (i + 1));
        wide.tag = sample;
        out[n] = readWide(wide, @truncate(sample));
        out[n + 1] = sumWide(wide);
        out[n + 2] = passOn(wide, @truncate(sample >> 3));
        n += 3;
    }
}

const Pair = struct { x: u32, y: u8, z: u8 };

/// Loads through a pointer that chooses between two locals, between two fields of one local, after
/// a write to one candidate, and in a loop. For each sample, writes four words.
pub fn local_pointer_choice(out: anytype) void {
    var n: usize = 0;
    for (samples) |sample| {
        const low: u32 = @truncate(sample);
        var a: u32 = low;
        var b: u32 = low ^ 0x5a5a5a5a;
        const chosen = if (sample & 1 != 0) &a else &b;
        out[n] = chosen.*;
        var pair: Pair = .{ .x = low +% 1, .y = @truncate(sample >> 8), .z = @truncate(sample >> 40) };
        const byte = if (sample >= 0x80000000) &pair.y else &pair.z;
        out[n + 1] = byte.*;
        var c: u32 = low >> 3;
        var d: u32 = low *% 3;
        const word = if (low > 0x1000) &c else &d;
        c +%= 5;
        out[n + 2] = word.*;
        var e: u32 = low | 1;
        var f: u32 = low >> 7;
        var cursor = &e;
        for (0..@as(usize, @truncate(sample % 3))) |_| cursor = if (cursor == &e) &f else &e;
        f +%= 7;
        out[n + 3] = cursor.*;
        n += 4;
    }
}

const digits = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };

const Entry = extern struct { word: u32, half: u16 };

const entries: [8]Entry = values: {
    var values: [8]Entry = undefined;
    for (&values, 0..) |*entry, i| entry.* = .{ .word = i * 77, .half = i * 3 };
    break :values values;
};

const big_table: [1100]u32 = values: {
    @setEvalBranchQuota(10000);
    var values: [1100]u32 = undefined;
    for (&values, 0..) |*value, i| value.* = @truncate(i *% 2654435761);
    break :values values;
};

/// Reads a 256-byte array after a store changes its memory, reads the copy of a local variable
/// after a store changes the variable, and reads comptime-known arrays at runtime indexes. For each
/// sample, writes seven words. The last 64 words of `out` hold the 256-byte array.
pub fn aggregate_loads(out: anytype) void {
    const memory = out[960..][0..64];
    var n: usize = 0;
    for (samples) |sample| {
        const index: usize = @truncate(sample % 64);
        for (memory, 0..) |*word, i| word.* = @truncate(sample *% (i + 3));
        const before = memory.*;
        memory[index] +%= 1;
        var before_sum: u32 = 0;
        for (before) |word| before_sum +%= word;
        out[n] = before_sum;
        out[n + 1] = before[3];
        var local = memory.*;
        const snapshot = local;
        local[index] = 7;
        var snapshot_sum: u32 = 0;
        for (snapshot) |word| snapshot_sum +%= word;
        out[n + 2] = snapshot_sum +% local[index];
        var after_sum: u32 = 0;
        for (memory.*) |word| after_sum +%= word;
        out[n + 3] = after_sum;
        out[n + 4] = digits[sample % digits.len];
        const entry = entries[sample % entries.len];
        out[n + 5] = entry.word +% entry.half;
        out[n + 6] = big_table[sample % big_table.len];
        n += 7;
    }
}

/// Chooses values of each register class with `if`, with nested choices, with a division that runs
/// only for a divisor that is not zero, and in a loop. For each sample, writes eight words.
pub fn value_choices(out: anytype) void {
    var n: usize = 0;
    var toggle: u32 = 0x9e3779b9;
    for (samples) |sample| {
        const low: u32 = @truncate(sample);
        const high: u32 = @truncate(sample >> 32);
        const odd = sample & 1 != 0;
        const large = low > 0x1000;
        const wide: u64 = if (odd) sample else ~sample;
        out[n] = @truncate(wide ^ (wide >> 32));
        const half: u16 = if (large) @truncate(low) else @truncate(high);
        out[n + 1] = half;
        const x: f32 = @floatFromInt(low >> 8);
        const y: f32 = if (odd) @mulAdd(f32, x, -2.0, 1.0) else 1.0;
        out[n + 2] = @bitCast(y);
        const z: f64 = if (!large) @as(f64, @floatFromInt(high)) * 0.5 else -3.25;
        out[n + 3] = @truncate(@as(u64, @bitCast(z)) >> 32);
        out[n + 4] = @intFromBool(if (odd) large else !large);
        const mixed = low ^ high;
        out[n + 5] = if (low < high) (if (high < mixed) high else mixed) else (if (low < mixed) low else mixed);
        out[n + 6] = if (high != 0) low / high else 0;
        toggle = if (toggle > low) toggle -% low else toggle +% high;
        out[n + 7] = toggle;
        n += 8;
    }
}

pub const names = [_][]const u8{
    "zero_extend",
    "float_div_trunc",
    "vector_u4",
    "vector_u14",
    "vector_memory",
    "counted_loops",
    "large_arguments",
    "local_pointer_choice",
    "aggregate_loads",
    "value_choices",
};
