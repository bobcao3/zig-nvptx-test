//! On a terminal, `rewind` moves the cursor up over the rows after the last `keep` or `rewind`, so
//! that the next rows replace them. A row sets its line style before it clears the line, so that
//! the line has the row background. Without color, `row` removes the `cell` styles from the format.

const std = @import("std");
const TerminalTableRenderer = @This();

writer: *std.Io.Writer,
terminal: bool,
color: bool,
light_background: bool,
live_rows: u32 = 0,
rows_since_heading: u32 = 0,

pub const Look = enum { title, heading, body, highlight };

pub const cell = struct {
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const italic = "\x1b[3m";
    pub const cyan = "\x1b[36m";
    /// Ends the styles above and keeps the background of the row.
    pub const normal = "\x1b[22;23;39m";
};

pub fn init(
    io: std.Io,
    file: std.Io.File,
    writer: *std.Io.Writer,
    environ: *const std.process.Environ.Map,
) !TerminalTableRenderer {
    const terminal = try file.supportsAnsiEscapeCodes(io);
    const no_color = environ.get("NO_COLOR") orelse "";
    const colors = environ.get("COLORFGBG") orelse "";
    const last = colors[if (std.mem.findScalarLast(u8, colors, ';')) |i| i + 1 else 0..];
    return .{
        .writer = writer,
        .terminal = terminal,
        .color = terminal and no_color.len == 0,
        .light_background = std.mem.eql(u8, last, "7") or std.mem.eql(u8, last, "15"),
    };
}

pub fn row(
    renderer: *TerminalTableRenderer,
    look: Look,
    comptime format: []const u8,
    args: anytype,
) !void {
    const writer = renderer.writer;
    if (renderer.color) try writer.writeAll(renderer.lineStyle(look));
    if (renderer.terminal) try writer.writeAll("\x1b[2K");
    if (renderer.color) {
        try writer.print(format ++ "\x1b[0m\n", args);
    } else {
        try writer.print(comptime plain(format) ++ "\n", args);
    }
    try writer.flush();
    renderer.live_rows += 1;
    renderer.rows_since_heading = if (look == .heading) 0 else renderer.rows_since_heading + 1;
}

pub fn keep(renderer: *TerminalTableRenderer) void {
    renderer.live_rows = 0;
}

pub fn rewind(renderer: *TerminalTableRenderer) !void {
    if (renderer.terminal and renderer.live_rows != 0) {
        try renderer.writer.print("\x1b[{d}A", .{renderer.live_rows});
        renderer.rows_since_heading -= renderer.live_rows;
    }
    renderer.live_rows = 0;
}

fn lineStyle(renderer: *const TerminalTableRenderer, look: Look) []const u8 {
    const light = renderer.light_background;
    return switch (look) {
        .title => "\x1b[0;1m",
        .heading => "\x1b[0;1;97;48;5;24m",
        .body => if (renderer.rows_since_heading % 2 == 0)
            "\x1b[0m"
        else if (light) "\x1b[0;48;5;254m" else "\x1b[0;48;5;236m",
        .highlight => if (light) "\x1b[0;48;5;194m" else "\x1b[0;48;5;22m",
    };
}

fn plain(comptime format: []const u8) []const u8 {
    comptime {
        var result: []const u8 = "";
        var rest = format;
        while (std.mem.findScalar(u8, rest, 0x1b)) |start| {
            result = result ++ rest[0..start];
            rest = rest[std.mem.findScalarPos(u8, rest, start, 'm').? + 1 ..];
        }
        return result ++ rest;
    }
}
