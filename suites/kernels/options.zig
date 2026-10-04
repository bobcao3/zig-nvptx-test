const Config = struct {
    name: []const u8,
    threads: u32,
    occupancy: u32,
    cluster: u32,
    registers: ?u32 = null,
};

fn Fill(comptime cfg: Config) type {
    return struct {
        fn kernel(out: [*]addrspace(.global) u32) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = cfg.threads },
            .min_work_groups_per_sm = cfg.occupancy,
            .cluster_size = .{ .x = cfg.cluster },
            .max_registers_per_work_item = cfg.registers,
        } }) void {
            out[@workGroupId(0) * cfg.threads + @workItemId(0)] = cfg.threads;
        }
        comptime {
            @export(&kernel, .{ .name = cfg.name });
        }
    };
}

comptime {
    _ = Fill(.{ .name = "fill128", .threads = 128, .occupancy = 4, .cluster = 2 });
    _ = Fill(.{ .name = "fill256", .threads = 256, .occupancy = 2, .cluster = 1, .registers = 128 });
}

export fn exact(out: [*]addrspace(.global) u32) callconv(.{ .nvptx_kernel = .{ .required_work_group_size = .{ .x = 32, .y = 4 } } }) void {
    out[@workItemId(1) * 32 + @workItemId(0)] = 1;
}
