//! Tests of the `cuda` module against a real GPU. Run with
//! `zig test --dep cuda -Mroot=cuda_test.zig -Mcuda=cuda.zig -lc`. The kernels are PTX text that
//! the driver compiles for the installed GPU.

const std = @import("std");
const cuda = @import("cuda");

const kernels: [:0]const u8 =
    \\.version 7.8
    \\.target sm_75
    \\.address_size 64
    \\
    \\.visible .entry readTexture(
    \\    .param .u64 texture, .param .u64 out, .param .u32 width, .param .s32 shift)
    \\{
    \\    .reg .b32 %r<8>;
    \\    .reg .f32 %f<9>;
    \\    .reg .b64 %rd<6>;
    \\    ld.param.u64 %rd1, [texture];
    \\    ld.param.u64 %rd2, [out];
    \\    ld.param.u32 %r1, [width];
    \\    ld.param.s32 %r2, [shift];
    \\    mov.u32 %r3, %tid.x;
    \\    mov.u32 %r4, %tid.y;
    \\    add.s32 %r5, %r3, %r2;
    \\    cvt.rn.f32.s32 %f5, %r5;
    \\    cvt.rn.f32.s32 %f6, %r4;
    \\    add.f32 %f7, %f5, 0f3F000000;
    \\    add.f32 %f8, %f6, 0f3F000000;
    \\    tex.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [%rd1, {%f7, %f8}];
    \\    mad.lo.u32 %r6, %r4, %r1, %r3;
    \\    mul.wide.u32 %rd3, %r6, 4;
    \\    add.u64 %rd4, %rd2, %rd3;
    \\    st.global.f32 [%rd4], %f1;
    \\    ret;
    \\}
    \\
    \\.visible .entry writeSurface(.param .u64 surface, .param .u32 width)
    \\{
    \\    .reg .b32 %r<8>;
    \\    .reg .b64 %rd<2>;
    \\    ld.param.u64 %rd1, [surface];
    \\    ld.param.u32 %r1, [width];
    \\    mov.u32 %r2, %tid.x;
    \\    mov.u32 %r3, %tid.y;
    \\    mad.lo.u32 %r4, %r3, %r1, %r2;
    \\    shl.b32 %r5, %r2, 2;
    \\    sust.b.2d.b32.trap [%rd1, {%r5, %r3}], {%r4};
    \\    ret;
    \\}
    \\
    \\.visible .entry storeUnlessNull(.param .u64 pointer)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<2>;
    \\    .reg .b64 %rd<2>;
    \\    ld.param.u64 %rd1, [pointer];
    \\    setp.eq.u64 %p1, %rd1, 0;
    \\    @%p1 bra DONE;
    \\    mov.u32 %r1, 42;
    \\    st.global.u32 [%rd1], %r1;
    \\DONE:
    \\    ret;
    \\}
    \\
    \\.visible .entry gridBarrier(.param .u64 counter, .param .u64 out, .param .u32 blocks)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<6>;
    \\    .reg .b64 %rd<6>;
    \\    ld.param.u64 %rd1, [counter];
    \\    ld.param.u64 %rd2, [out];
    \\    ld.param.u32 %r4, [blocks];
    \\    mov.u32 %r1, %ctaid.x;
    \\    atom.global.add.u32 %r2, [%rd1], 1;
    \\SPIN:
    \\    ld.volatile.global.u32 %r3, [%rd1];
    \\    setp.lt.u32 %p1, %r3, %r4;
    \\    @%p1 bra SPIN;
    \\    mul.wide.u32 %rd3, %r1, 4;
    \\    add.u64 %rd4, %rd2, %rd3;
    \\    st.global.u32 [%rd4], %r3;
    \\    ret;
    \\}
;

const width = 8;
const height = 4;

fn expectTexelsEqual(actual: []const f32, expected: []const f32) !void {
    try std.testing.expectEqualSlices(f32, expected, actual);
}

fn sourceTexels() [width * height]f32 {
    var texels: [width * height]f32 = undefined;
    for (&texels, 0..) |*texel, index| texel.* = @floatFromInt(index * 3 + 1);
    return texels;
}

test "device queries" {
    const device = try cuda.init(0);
    defer device.deinit();

    try std.testing.expect(try device.totalMemory() > 1 << 20);
    try std.testing.expect(try device.attribute(.max_grid_dim_x) >= 65535);
    try std.testing.expect(try device.attribute(.max_grid_dim_y) >= 65535);
    try std.testing.expect(try device.attribute(.max_grid_dim_z) >= 65535);
    try std.testing.expect(try device.attribute(.compute_mode) >= 0);
    try std.testing.expect(try device.attribute(.managed_memory) >= 0);
    try std.testing.expect(try device.attribute(.pageable_memory_access) >= 0);
    try std.testing.expect(try device.attribute(.concurrent_managed_access) >= 0);
    try std.testing.expect(try device.attribute(.warp_size) == 32);

    const capability = try device.computeCapability();
    try std.testing.expect(cuda.coresPerMultiprocessor(capability.major, capability.minor) > 0);
    try std.testing.expectEqual(@as(i32, 128), cuda.coresPerMultiprocessor(8, 6));
}

test "occupancy, function attributes and limits" {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(kernels);
    defer module.unload();
    const function = try module.function("storeUnlessNull");

    const potential = try function.maxPotentialBlockSize(0, 0);
    try std.testing.expect(potential.block_size > 0 and potential.min_grid_size > 0);
    const blocks = try function.maxActiveBlocksPerMultiprocessor(potential.block_size, 0);
    try std.testing.expect(blocks >= 1);

    try std.testing.expect(try function.attribute(.num_regs) > 0);
    try std.testing.expect(try function.attribute(.max_threads_per_block) >= potential.block_size);
    try function.setMaxDynamicSharedMemory(48 * 1024);
    try std.testing.expectEqual(@as(i32, 48 * 1024), try function.attribute(.max_dynamic_shared_size_bytes));
    try function.setCacheConfig(.prefer_l1);

    try cuda.setLimit(.stack_size, 4096);
    try std.testing.expect(try cuda.getLimit(.stack_size) >= 4096);
    try cuda.setLimit(.malloc_heap_size, 16 << 20);
    try std.testing.expect(try cuda.getLimit(.malloc_heap_size) >= 16 << 20);
    try std.testing.expect(try cuda.getLimit(.printf_fifo_size) > 0);
}

test "memset and null pointer argument" {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(kernels);
    defer module.unload();
    const store = try module.function("storeUnlessNull");

    const buffer = try cuda.alloc(u32, 16);
    defer buffer.free();
    var host: [16]u32 = undefined;

    try buffer.memset32(0xdeadbeef);
    try buffer.copyTo(&host);
    for (host) |word| try std.testing.expectEqual(@as(u32, 0xdeadbeef), word);

    const stream = try cuda.Stream.create();
    defer stream.destroy();
    try buffer.memsetAsync(0x11, stream);
    try stream.synchronize();
    try buffer.copyTo(&host);
    for (host) |word| try std.testing.expectEqual(@as(u32, 0x11111111), word);

    try store.launch(.{ .grid = .{}, .block = .{} }, .{null});
    try store.launch(.{ .grid = .{}, .block = .{} }, .{@as(?cuda.Buffer(u32), null)});
    try cuda.synchronize();
    try store.launch(.{ .grid = .{}, .block = .{} }, .{@as(?cuda.Buffer(u32), buffer)});
    try cuda.synchronize();
    try buffer.copyTo(&host);
    try std.testing.expectEqual(@as(u32, 42), host[0]);
    try std.testing.expectEqual(@as(u32, 0x11111111), host[1]);
}

test "texture object over a two-dimensional array" {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(kernels);
    defer module.unload();
    const read_texture = try module.function("readTexture");

    const array = try cuda.Array.create(.{ .width = width, .height = height, .format = .float, .channels = 1 });
    defer array.destroy();
    const source = sourceTexels();
    try array.copyFromHost(f32, &source);

    var round_trip: [width * height]f32 = undefined;
    try array.copyToHost(f32, &round_trip);
    try expectTexelsEqual(&round_trip, &source);

    const out = try cuda.alloc(f32, width * height);
    defer out.free();
    var result: [width * height]f32 = undefined;
    const config: cuda.LaunchConfig = .{ .grid = .{}, .block = .{ .x = width, .y = height } };

    const texture = try cuda.TextureObject.create(.{ .array = array }, .{});
    defer texture.destroy();
    try read_texture.launch(config, .{ texture, out, @as(u32, width), @as(i32, 0) });
    try out.copyTo(&result);
    try expectTexelsEqual(&result, &source);

    try read_texture.launch(config, .{ texture, out, @as(u32, width), @as(i32, 3) });
    try out.copyTo(&result);
    for (0..height) |y| {
        for (0..width) |x| {
            const clamped = @min(x + 3, width - 1);
            try std.testing.expectEqual(source[y * width + clamped], result[y * width + x]);
        }
    }

    const border = try cuda.TextureObject.create(.{ .array = array }, .{
        .address_mode = @splat(.border),
        .border_color = @splat(-1),
    });
    defer border.destroy();
    try read_texture.launch(config, .{ border, out, @as(u32, width), @as(i32, 3) });
    try out.copyTo(&result);
    try std.testing.expectEqual(@as(f32, -1), result[width - 1]);
    try std.testing.expectEqual(source[3], result[0]);
}

test "texture object over pitch-linear memory" {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(kernels);
    defer module.unload();
    const read_texture = try module.function("readTexture");

    const alignment: usize = @intCast(try device.attribute(.texture_pitch_alignment));
    const pitch_bytes = std.mem.alignForward(usize, width * @sizeOf(f32), alignment);
    const pitch = pitch_bytes / @sizeOf(f32);
    var padded: [height * 64]f32 = @splat(0);
    std.debug.assert(pitch <= 64);
    const source = sourceTexels();
    for (0..height) |y| @memcpy(padded[y * pitch ..][0..width], source[y * width ..][0..width]);

    const memory = try cuda.dupe(f32, padded[0 .. height * pitch]);
    defer memory.free();
    const texture = try cuda.TextureObject.create(.{ .pitch2d = .{
        .address = memory.address,
        .format = .float,
        .channels = 1,
        .width = width,
        .height = height,
        .pitch_bytes = pitch_bytes,
    } }, .{});
    defer texture.destroy();

    const out = try cuda.alloc(f32, width * height);
    defer out.free();
    try read_texture.launch(
        .{ .grid = .{}, .block = .{ .x = width, .y = height } },
        .{ texture, out, @as(u32, width), @as(i32, 0) },
    );
    var result: [width * height]f32 = undefined;
    try out.copyTo(&result);
    try expectTexelsEqual(&result, &source);

    const array = try cuda.Array.create(.{ .width = width, .height = height, .format = .float, .channels = 1 });
    defer array.destroy();
    try array.copyFromBuffer(f32, memory, pitch_bytes);
    const out_memory = try cuda.alloc(f32, height * pitch);
    defer out_memory.free();
    try array.copyToBuffer(f32, out_memory, pitch_bytes);
    var padded_result: [height * 64]f32 = undefined;
    try out_memory.copyTo(padded_result[0 .. height * pitch]);
    for (0..height) |y| {
        try expectTexelsEqual(padded_result[y * pitch ..][0..width], source[y * width ..][0..width]);
    }
}

test "surface object writes a two-dimensional array" {
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(kernels);
    defer module.unload();
    const write_surface = try module.function("writeSurface");

    const array = try cuda.Array.create3d(.{
        .width = width,
        .height = height,
        .format = .unsigned_int32,
        .channels = 1,
        .flags = .{ .surface_ldst = true },
    });
    defer array.destroy();
    const surface = try cuda.SurfaceObject.create(.{ .array = array });
    defer surface.destroy();

    try write_surface.launch(
        .{ .grid = .{}, .block = .{ .x = width, .y = height } },
        .{ surface, @as(u32, width) },
    );
    var result: [width * height]u32 = undefined;
    try array.copyToHost(u32, &result);
    for (result, 0..) |value, index| try std.testing.expectEqual(@as(u32, @intCast(index)), value);
}

test "layered and cubemap arrays" {
    const device = try cuda.init(0);
    defer device.deinit();

    const layers = 3;
    const layered = try cuda.Array.create3d(.{
        .width = width,
        .height = height,
        .depth = layers,
        .format = .unsigned_int16,
        .channels = 2,
        .flags = .{ .layered = true },
    });
    defer layered.destroy();
    var layered_source: [width * height * layers * 2]u16 = undefined;
    for (&layered_source, 0..) |*element, index| element.* = @intCast(index * 7);
    try layered.copyFromHost(u16, &layered_source);
    var layered_result: [layered_source.len]u16 = undefined;
    try layered.copyToHost(u16, &layered_result);
    try std.testing.expectEqualSlices(u16, &layered_source, &layered_result);
    const layered_texture = try cuda.TextureObject.create(.{ .array = layered }, .{
        .filter_mode = .linear,
        .read_mode = .normalized_float,
        .normalized_coordinates = true,
    });
    layered_texture.destroy();

    const edge = 4;
    const cubemap = try cuda.Array.create3d(.{
        .width = edge,
        .height = edge,
        .depth = 6,
        .format = .float,
        .channels = 1,
        .flags = .{ .cubemap = true },
    });
    defer cubemap.destroy();
    var cube_source: [edge * edge * 6]f32 = undefined;
    for (&cube_source, 0..) |*element, index| element.* = @floatFromInt(index);
    try cubemap.copyFromHost(f32, &cube_source);
    var cube_result: [cube_source.len]f32 = undefined;
    try cubemap.copyToHost(f32, &cube_result);
    try std.testing.expectEqualSlices(f32, &cube_source, &cube_result);
    const cube_texture = try cuda.TextureObject.create(.{ .array = cubemap }, .{
        .filter_mode = .linear,
        .seamless_cubemap = true,
        .normalized_coordinates = true,
    });
    cube_texture.destroy();

    const line = try cuda.Array.create(.{ .width = 16, .format = .signed_int8, .channels = 4 });
    defer line.destroy();
    var line_source: [64]i8 = undefined;
    for (&line_source, 0..) |*element, index| element.* = @intCast(index);
    try line.copyFromHost(i8, &line_source);
    var line_result: [64]i8 = undefined;
    try line.copyToHost(i8, &line_result);
    try std.testing.expectEqualSlices(i8, &line_source, &line_result);
}

test "cooperative launch" {
    const device = try cuda.init(0);
    defer device.deinit();
    if (try device.attribute(.cooperative_launch) == 0) return error.SkipZigTest;
    const module = try cuda.Module.load(kernels);
    defer module.unload();
    const barrier = try module.function("gridBarrier");

    const blocks: u32 = @intCast(try device.attribute(.multiprocessor_count));
    const counter = try cuda.alloc(u32, 1);
    defer counter.free();
    try counter.memset(0);
    const out = try cuda.alloc(u32, blocks);
    defer out.free();

    try barrier.launchCooperative(.{ .grid = .{ .x = blocks }, .block = .{} }, .{ counter, out, blocks });
    try cuda.synchronize();
    const result = try std.testing.allocator.alloc(u32, blocks);
    defer std.testing.allocator.free(result);
    try out.copyTo(result);
    for (result) |seen| try std.testing.expectEqual(blocks, seen);

    try counter.memset(0);
    try barrier.launch(
        .{ .grid = .{ .x = blocks }, .block = .{}, .cooperative = true },
        .{ counter, out, blocks },
    );
    try cuda.synchronize();
}
