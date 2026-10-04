//! Renders the final scene of "Ray Tracing in One Weekend" (https://raytracing.github.io) on the
//! GPU and writes it to `image.png`. The book writes a PPM image instead.

const std = @import("std");
const cuda = @import("cuda");
const scene = @import("scene");
const png = @import("png.zig");

const samples_per_pixel = 500;
const max_depth = 50;
/// The kernels render this many samples per launch, so that the host can report progress.
const samples_per_launch = 10;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("render.ptx"));
    defer module.unload();
    const render = try module.function("render");
    const resolve = try module.function("resolve");

    var prng: std.Random.DefaultPrng = .init(0);
    const world = try finalScene(gpa, prng.random());
    defer gpa.free(world);
    const camera: scene.Camera = .init(.{
        .aspect_ratio = 16.0 / 9.0,
        .image_width = 1200,
        .vfov = 20,
        .lookfrom = .init(13, 2, 3),
        .lookat = .init(0, 0, 0),
        .vup = .init(0, 1, 0),
        .defocus_angle = 0.6,
        .focus_dist = 10,
    });
    const pixel_count = camera.image_width * camera.image_height;

    const spheres = try cuda.dupe(scene.Sphere, world);
    defer spheres.free();
    const accumulator = try cuda.alloc(scene.Vec3, pixel_count);
    defer accumulator.free();
    try accumulator.memset(0);
    const pixels = try cuda.alloc(scene.Rgba8, pixel_count);
    defer pixels.free();

    var done: u32 = 0;
    while (done < samples_per_pixel) : (done += samples_per_launch) {
        try render.launch(.{
            .grid = .{ .x = (camera.image_width + 15) / 16, .y = (camera.image_height + 15) / 16 },
            .block = .{ .x = 16, .y = 16 },
        }, .{
            accumulator,
            spheres,
            @as(u32, @intCast(world.len)),
            camera,
            @as(u32, max_depth),
            done,
            @as(u32, samples_per_launch),
        });
        try cuda.synchronize();
        std.log.info("Samples remaining: {d}", .{samples_per_pixel - done - samples_per_launch});
    }
    try resolve.launch(.{
        .grid = .{ .x = (pixel_count + 255) / 256 },
        .block = .{ .x = 256 },
    }, .{ accumulator, pixels, pixel_count, @as(u32, samples_per_pixel) });

    const image = try gpa.alloc(scene.Rgba8, pixel_count);
    defer gpa.free(image);
    try pixels.copyTo(image);
    const file = try png.encode(gpa, camera.image_width, camera.image_height, image);
    defer gpa.free(file);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = "image.png", .data = file });
    std.log.info("Done.", .{});
}

/// Builds the scene of section 14.1 of the book.
fn finalScene(gpa: std.mem.Allocator, random: std.Random) ![]scene.Sphere {
    var world: std.ArrayList(scene.Sphere) = .empty;
    errdefer world.deinit(gpa);
    try world.append(gpa, .{
        .center = .init(0, -1000, 0),
        .radius = 1000,
        .material = .lambertian(.init(0.5, 0.5, 0.5)),
    });
    for (0..22) |i| {
        for (0..22) |j| {
            const a = @as(f32, @floatFromInt(i)) - 11;
            const b = @as(f32, @floatFromInt(j)) - 11;
            const choose_material = random.float(f32);
            const x = a + 0.9 * random.float(f32);
            const center: scene.Vec3 = .init(x, 0.2, b + 0.9 * random.float(f32));
            const offset = center.vector() - scene.Vec3.init(4, 0.2, 0).vector();
            if (@sqrt(scene.dot(offset, offset)) <= 0.9) continue;
            const material: scene.Material = if (choose_material < 0.8)
                .lambertian(.fromVector(randomColor(random, 0, 1) * randomColor(random, 0, 1)))
            else if (choose_material < 0.95)
                .metal(.fromVector(randomColor(random, 0.5, 1)), 0.5 * random.float(f32))
            else
                .dielectric(1.5);
            try world.append(gpa, .{ .center = center, .radius = 0.2, .material = material });
        }
    }
    try world.appendSlice(gpa, &.{
        .{ .center = .init(0, 1, 0), .radius = 1, .material = .dielectric(1.5) },
        .{ .center = .init(-4, 1, 0), .radius = 1, .material = .lambertian(.init(0.4, 0.2, 0.1)) },
        .{ .center = .init(4, 1, 0), .radius = 1, .material = .metal(.init(0.7, 0.6, 0.5), 0) },
    });
    return world.toOwnedSlice(gpa);
}

fn randomColor(random: std.Random, min: f32, max: f32) @Vector(3, f32) {
    return .{
        min + (max - min) * random.float(f32),
        min + (max - min) * random.float(f32),
        min + (max - min) * random.float(f32),
    };
}
