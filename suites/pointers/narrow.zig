//! Each kernel checks a rule of the 32-bit pointers into `.shared` and `.local` memory, whose null
//! pointers hold the all-ones address. A failed check stores its line number in `out[0]`.

const nv = @import("std").nvptx;

const Out = [*]addrspace(.global) u32;

var words: [64]u32 addrspace(.shared) = undefined;
var pairs: [4]Pair addrspace(.shared) = undefined;

const Pair = extern struct { first: *addrspace(.shared) u32, count: u32 };

fn expect(out: Out, line: u32, ok: bool) void {
    if (!ok) _ = @cmpxchgStrong(u32, &out[0], 0, line, .monotonic, .monotonic);
}

noinline fn pick(i: u32) ?*addrspace(.shared) u32 {
    return if (i < words.len) &words[i] else null;
}

export fn optional_shared(out: Out) callconv(.kernel) void {
    const i = @workItemId(0);
    const inside = pick(i % 64);
    const outside = pick(i + 64);
    expect(out, @src().line, inside != null);
    expect(out, @src().line, outside == null);
    expect(out, @src().line, @intFromPtr(outside) == 0xffff_ffff);
    expect(out, @src().line, @sizeOf(?*addrspace(.shared) u32) == 4);
    const generic_outside: ?*u32 = @addrSpaceCast(outside);
    expect(out, @src().line, generic_outside == null);
    const generic_inside: ?*u32 = @addrSpaceCast(inside);
    expect(out, @src().line, generic_inside != null);
    const back: ?*addrspace(.shared) u32 = @addrSpaceCast(generic_inside);
    expect(out, @src().line, back == inside);
    const generic_null: ?*u32 = if (i < 1024) null else generic_inside;
    const shared_null: ?*addrspace(.shared) u32 = @addrSpaceCast(generic_null);
    expect(out, @src().line, shared_null == null);
}

export fn pointer_fields(out: Out) callconv(.kernel) void {
    const i = @workItemId(0) % 64;
    words[i] = i * 3;
    nv.barrierSyncAligned(.work_group, null);
    if (i < pairs.len) pairs[i] = .{ .first = &words[i * 16], .count = 16 };
    nv.barrierSyncAligned(.work_group, null);
    const pair = pairs[i % 4];
    expect(out, @src().line, @sizeOf(Pair) == 8);
    expect(out, @src().line, pair.first.* == (i % 4) * 16 * 3);
    const many: [*]addrspace(.shared) u32 = @ptrCast(pair.first);
    expect(out, @src().line, many[pair.count - 1] == ((i % 4) * 16 + 15) * 3);
    const slice: []addrspace(.shared) u32 = many[0..pair.count];
    expect(out, @src().line, slice[i % 16] == ((i % 4) * 16 + i % 16) * 3);
    expect(out, @src().line, @intFromPtr(&words[i]) - @intFromPtr(&words[0]) == i * 4);
    const address = @intFromPtr(&words[i]);
    expect(out, @src().line, address <= 0xffff_ffff);
    const round_trip: *addrspace(.shared) u32 = @ptrFromInt(address);
    expect(out, @src().line, round_trip == &words[i]);
}

export fn local_pointers(out: Out) callconv(.kernel) void {
    const i = @workItemId(0);
    var buffer: [8]u32 = undefined;
    for (&buffer, 0..) |*word, k| word.* = @intCast(k * 7 + i);
    const local: *addrspace(.local) [8]u32 = @addrSpaceCast(&buffer);
    expect(out, @src().line, @sizeOf(@TypeOf(local)) == 4);
    expect(out, @src().line, local[i % 8] == (i % 8) * 7 + i);
    local[(i + 1) % 8] += 1;
    expect(out, @src().line, buffer[(i + 1) % 8] == ((i + 1) % 8) * 7 + i + 1);
}
