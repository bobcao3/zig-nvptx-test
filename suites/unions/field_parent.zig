const h = @import("harness.zig");

const Node = struct {
    id: u32,
    wide: u64,
    link: Link,
    tail: u8,
};

const Link = struct { next: ?*Link = null, value: u32 };

const Extern = extern struct { a: u8, b: u32, c: u16 };

const Choice = union {
    small: u8,
    word: u32,
    pair: [2]u32,
};

const Tagged = union(enum) { x: u32, y: u64 };

noinline fn nodeOf(link: *Link) *Node {
    return @fieldParentPtr("link", link);
}

noinline fn externOfB(p: *u32) *Extern {
    return @alignCast(@fieldParentPtr("b", p));
}

noinline fn externOfC(p: *u16) *Extern {
    return @alignCast(@fieldParentPtr("c", p));
}

noinline fn choiceOf(p: *u32) *Choice {
    return @fieldParentPtr("word", p);
}

noinline fn taggedOf(p: *u64) *Tagged {
    return @fieldParentPtr("y", p);
}

fn run(out: h.Out) void {
    const t = h.tid();
    var k: u32 = 0;
    var nodes = [2]Node{
        .{ .id = 11, .wide = 0x1111, .link = .{ .value = 100 }, .tail = 1 },
        .{ .id = 22, .wide = 0x2222, .link = .{ .value = 200 }, .tail = 2 },
    };
    const n = nodeOf(&nodes[t].link);
    out[k] = n.id;
    k +%= 1;
    out[k] = @truncate(n.wide);
    k +%= 1;
    out[k] = n.tail;
    k +%= 1;
    nodes[0].link.next = &nodes[1].link;
    out[k] = nodeOf(nodes[t].link.next.?).id;
    k +%= 1;
    const link_ptr: *Link = &nodes[1 +% t].link;
    out[k] = @as(*Node, @fieldParentPtr("link", link_ptr)).tail;
    k +%= 1;

    var exts = [2]Extern{ .{ .a = 1, .b = 2, .c = 3 }, .{ .a = 4, .b = 5, .c = 6 } };
    out[k] = externOfB(&exts[1 -% t].b).a;
    k +%= 1;
    out[k] = externOfC(&exts[t].c).b;
    k +%= 1;
    externOfB(&exts[t].b).c = 77;
    out[k] = exts[0].c;
    k +%= 1;

    var choice: Choice = .{ .word = 0xabcdef };
    out[k] = choiceOf(&choice.word).word;
    k +%= 1;
    var tagged: Tagged = .{ .y = 0x123456789 +% @as(u64, t) };
    const y = taggedOf(&tagged.y);
    out[k] = @intFromEnum(y.*);
    k +%= 1;
    out[k] = @truncate(y.y);
    k +%= 1;
}

pub const H = h.Harness(run, 11);
comptime {
    _ = H;
}
