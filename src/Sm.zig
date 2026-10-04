//! A PTX target architecture such as `sm_80`, `sm_90a` or `sm_120f`.

const std = @import("std");
const Sm = @This();

major: u32,
minor: u32,
kind: Kind,

/// The PTX ISA, section "Target architectures", defines these loading rules: a base target runs
/// on every device of an equal or newer compute capability, a family target on the devices of its
/// major version with an equal or newer minor version, and an arch-specific target only on the
/// device of its own compute capability.
pub const Kind = enum { base, family, arch_specific };

pub fn parse(name: []const u8) ?Sm {
    var digits = std.mem.cutPrefix(u8, name, "sm_") orelse return null;
    var kind: Kind = .base;
    if (std.mem.cutSuffix(u8, digits, "a")) |rest| {
        digits = rest;
        kind = .arch_specific;
    } else if (std.mem.cutSuffix(u8, digits, "f")) |rest| {
        digits = rest;
        kind = .family;
    }
    const number = std.fmt.parseUnsigned(u32, digits, 10) catch return null;
    if (number < 10) return null;
    return .{ .major = number / 10, .minor = number % 10, .kind = kind };
}

/// Returns the target of the first `.target` directive of the PTX text `ptx`.
pub fn fromPtx(ptx: []const u8) ?Sm {
    const directive = ".target ";
    const start = (std.mem.find(u8, ptx, directive) orelse return null) + directive.len;
    const end = std.mem.findAnyPos(u8, ptx, start, ", \t\r\n") orelse ptx.len;
    return parse(ptx[start..end]);
}

pub fn loadsOn(sm: Sm, device_major: u32, device_minor: u32) bool {
    return switch (sm.kind) {
        .base => device_major * 10 + device_minor >= sm.major * 10 + sm.minor,
        .family => device_major == sm.major and device_minor >= sm.minor,
        .arch_specific => device_major == sm.major and device_minor == sm.minor,
    };
}

pub fn format(sm: Sm, w: *std.Io.Writer) std.Io.Writer.Error!void {
    try w.print("sm_{d}{d}{s}", .{ sm.major, sm.minor, switch (sm.kind) {
        .base => "",
        .family => "f",
        .arch_specific => "a",
    } });
}

test "loading rules" {
    const device_12_1 = .{ 12, 1 };
    inline for (.{
        .{ "sm_80", true },
        .{ "sm_121", true },
        .{ "sm_121a", true },
        .{ "sm_120f", true },
        .{ "sm_121f", true },
        .{ "sm_120a", false },
        .{ "sm_90a", false },
        .{ "sm_100f", false },
        .{ "sm_130", false },
    }) |case| {
        try std.testing.expectEqual(case[1], parse(case[0]).?.loadsOn(device_12_1[0], device_12_1[1]));
    }
    try std.testing.expectFmt("sm_100f", "{f}", .{parse("sm_100f").?});
}
