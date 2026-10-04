const h = @import("harness.zig");

const Kind = enum(u8) { a, b, c, d };
const Marker = union(Kind) { a, b, c, d };

const Wide = enum(u32) { p = 10, q = 20, r = 30 };
const WideMarker = union(Wide) { p, q, r };

const Single = union(enum) { only: u32 };
const OneByte = union(enum) { only: u8 };
const Nothing = union(enum) { only };
const Mixed = union(enum(u16)) { none: void, small: u8, big: [5]u16 = 300 };

fn markerValue(m: Marker) u32 {
    return switch (m) {
        .a => 1,
        .b => 20,
        .c => 300,
        .d => 4000,
    };
}

noinline fn wideValue(m: WideMarker) u32 {
    return switch (m) {
        .p => 5,
        .q => 6,
        .r => 7,
    };
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    const kind: Kind = @enumFromInt(t +% 2);
    const m: Marker = kind;
    out[k] = markerValue(m);
    k +%= 1;
    out[k] = @intFromEnum(@as(Kind, m));
    k +%= 1;
    out[k] = @intFromBool(m == .c);
    k +%= 1;
    const w: Wide = if (t == 0) .q else .r;
    const wm: WideMarker = w;
    out[k] = wideValue(wm);
    k +%= 1;
    out[k] = @intFromEnum(@as(Wide, wm));
    k +%= 1;

    const s: Single = .{ .only = 99 +% t };
    out[k] = switch (s) {
        .only => |v| v,
    };
    k +%= 1;
    const ob: OneByte = .{ .only = @truncate(250 +% t) };
    out[k] = ob.only;
    k +%= 1;
    const nothing: Nothing = .only;
    out[k] = @intFromBool(nothing == .only);
    k +%= 1;

    var mixed: Mixed = if (t == 0) .{ .small = 17 } else .none;
    out[k] = switch (mixed) {
        .none => 0,
        .small => |v| v,
        .big => |v| v[0],
    };
    k +%= 1;
    out[k] = @intFromEnum(@as(@typeInfo(Mixed).@"union".tag_type.?, mixed));
    k +%= 1;
    mixed = .{ .big = .{ 1, 2, 3, 4, 5 +% @as(u16, @truncate(t)) } };
    out[k] = switch (mixed) {
        .none => 0,
        .small => |v| v,
        .big => |v| v[4],
    };
    k +%= 1;
    out[k] = @intFromEnum(@as(@typeInfo(Mixed).@"union".tag_type.?, mixed));
    k +%= 1;
}

pub const H = h.Harness(run, 12);
comptime {
    _ = H;
}
