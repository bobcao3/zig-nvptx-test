//! Each function keeps its local variables in a form that the PTX backend once put in local memory.
//! The functions run unchanged on the host and on the GPU, and the build checks that ptxas gives
//! the kernels of `locals_kernels.zig` no stack frame and no spilled registers.

const samples = [_]u64{
    0,          1,                  0x7fffffff,         0x80000000,
    0xffffffff, 0x123456789abcdef0, 0xfedcba9876543210, 0xffffffffffffffff,
};

/// The host and the launcher fill `out` with 0xdeadbeef, so `out[seed_index]` is a word that the
/// compiler cannot know.
const seed_index = 1023;

inline fn operand(out: anytype, sample: u64) u32 {
    return @as(u32, @truncate(sample)) ^ out[seed_index];
}

const primes = [_]u32{ 2, 3, 5, 7, 11, 13, 17, 19 };

pub fn constant_table(out: anytype) void {
    const digits: [16]u32 = .{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        out[n] = digits[x % digits.len] *% 100 +% primes[x / 16 % primes.len];
    }
}

pub fn element_pointers(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var rows: [4][8]u32 = undefined;
        inline for (&rows, 0..) |*row, i| {
            inline for (row, 0..) |*value, j| value.* = x *% @as(u32, i * 8 + j + 1);
        }
        var vectors: [4]@Vector(4, u32) = @splat(@splat(1));
        inline for (&vectors, rows) |*vector, row| {
            inline for (0..4) |j| vector.*[j] +%= row[j] ^ row[j + 4];
        }
        var sum: u32 = 0;
        inline for (rows, vectors) |row, vector| {
            inline for (row) |value| sum +%= value;
            inline for (0..4) |j| sum = sum *% 31 +% vector[j];
        }
        out[n] = sum;
    }
}

const Cursor = struct {
    index: u32,
    step: u32,

    inline fn advance(cursor: *Cursor) void {
        cursor.index +%= cursor.step;
    }
};

pub fn pointer_receiver(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var cursor: Cursor = .{ .index = x, .step = x >> 5 | 1 };
        for (0..x % 5) |_| cursor.advance();
        out[n] = cursor.index;
    }
}

const Tally = struct {
    sums: [4]u32,
    counts: [4]u32,

    inline fn add(tally: *Tally, word: u32) void {
        inline for (&tally.sums, &tally.counts, 0..) |*sum, *count, i| {
            const byte = word >> 8 * i & 0xff;
            sum.* +%= byte;
            count.* += @intFromBool(byte != 0);
        }
    }
};

pub fn array_fields(out: anytype) void {
    for (samples, 0..) |sample, n| {
        var tally: Tally = .{ .sums = @splat(0), .counts = @splat(0) };
        var word = operand(out, sample);
        for (0..4) |_| {
            tally.add(word);
            word = word *% 2654435761 +% 1;
        }
        var result: u32 = 0;
        inline for (tally.sums, tally.counts) |sum, count| result = result *% 31 +% (sum << 3 | count);
        out[n] = result;
    }
}

pub fn pointer_select(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        const first = out[2 * n ..];
        const second = out[2 * n + 1 ..];
        first[0] = x;
        second[0] = ~x;
        (if (x & 1 != 0) first else second)[0] = x >> 1;
    }
}

/// Swaps the bytes of two values in a loop, as `std.mem.swap` does at run time.
inline fn swapBytes(comptime T: type, a: *T, b: *T) void {
    const a_bytes: []align(@alignOf(T)) u8 = @ptrCast(a);
    const b_bytes: []align(@alignOf(T)) u8 = @ptrCast(b);
    for (a_bytes, b_bytes) |*a_byte, *b_byte| {
        const byte = a_byte.*;
        a_byte.* = b_byte.*;
        b_byte.* = byte;
    }
}

pub fn swapped_halves(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var low: u16 = @truncate(x);
        var high: u16 = @truncate(x >> 16);
        if (low < high) swapBytes(u16, &low, &high);
        out[n] = @as(u32, high) << 16 | low;
    }
}

pub fn swapped_words(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var a: u64 = sample ^ x;
        var b: u64 = @as(u64, x) << 32 | ~x;
        if (x & 1 != 0) swapBytes(u64, &a, &b);
        var c: f32 = @bitCast(x);
        var d: f32 = @bitCast(x *% 2654435761);
        if (x & 2 != 0) swapBytes(f32, &c, &d);
        out[n] = @as(u32, @truncate(a ^ b >> 3)) +% @as(u32, @bitCast(c)) *% 3 +% @as(u32, @bitCast(d));
    }
}

pub fn indexed_bytes(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var word: u32 = x;
        const bytes: *[4]u8 = @ptrCast(&word);
        bytes[x % 4] = @truncate(x >> 8);
        bytes[(x >> 2) % 4] +%= 1;
        const signed: *const [4]i8 = @ptrCast(&word);
        out[n] = word +% @as(u32, @bitCast(@as(i32, signed[(x >> 4) % 4])));
    }
}

pub fn indexed_triples(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var triple: [3]u8 = .{ @truncate(x), @truncate(x >> 8), @truncate(x >> 16) };
        var other: [3]u8 = .{ @truncate(x >> 24), @truncate(x >> 3), @truncate(x >> 11) };
        swapBytes([3]u8, &triple, &other);
        triple[x % 3] ^= other[(x >> 5) % 3];
        out[n] = @as(u32, triple[0]) | @as(u32, triple[1]) << 8 | @as(u32, triple[2]) << 16 | @as(u32, other[x % 3]) << 24;
    }
}

pub fn indexed_halves(out: anytype) void {
    for (samples, 0..) |sample, n| {
        const x = operand(out, sample);
        var halves: [3]u16 = .{ @truncate(x), @truncate(x >> 16), @truncate(sample >> 32) };
        halves[x % 3] = @truncate(x *% 2654435761);
        var pair: [2]f32 = .{ @bitCast(x), @bitCast(~x) };
        pair[(x >> 1) % 2] = @bitCast(x >> 3);
        out[n] = @as(u32, halves[(x >> 2) % 3]) +% @as(u32, @bitCast(pair[x % 2])) *% 7 +% halves[0];
    }
}

pub const names = [_][]const u8{
    "constant_table",
    "element_pointers",
    "pointer_receiver",
    "array_fields",
    "pointer_select",
    "swapped_halves",
    "swapped_words",
    "indexed_bytes",
    "indexed_triples",
    "indexed_halves",
};
