//! The kernels of the ray tracer. Each work item traces the paths of one pixel, as chapters 4 to 13
//! of "Ray Tracing in One Weekend" describe, and adds their colors to the accumulator. A loop
//! replaces the recursion of `ray_color`, a sphere array replaces `hittable_list`, and the tag of a
//! material replaces the material classes, because kernels have no virtual calls.

const std = @import("std");
const scene = @import("scene");

const Vec = @Vector(3, f32);
const splat = scene.splat;
const dot = scene.dot;
const unit = scene.unit;

/// The PCG hash of Jarzynski and Olano, "Hash Functions for GPU Rendering" (Journal of Computer
/// Graphics Techniques, 2020), applied to a state that each call advances.
const Rng = struct {
    state: u32,

    fn next(rng: *Rng) u32 {
        rng.state = rng.state *% 747796405 +% 2891336453;
        const word = ((rng.state >> @intCast((rng.state >> 28) + 4)) ^ rng.state) *% 277803737;
        return (word >> 22) ^ word;
    }

    fn float(rng: *Rng) f32 {
        return @as(f32, @floatFromInt(rng.next() >> 8)) * 0x1p-24;
    }

    fn range(rng: *Rng, min: f32, max: f32) f32 {
        return min + (max - min) * rng.float();
    }

    fn unitVector(rng: *Rng) Vec {
        while (true) {
            const p: Vec = .{ rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1) };
            const length_squared = dot(p, p);
            // The book uses 1e-160, which `f32` cannot hold.
            if (1e-30 < length_squared and length_squared <= 1) {
                return p / splat(@sqrt(length_squared));
            }
        }
    }

    fn inUnitDisk(rng: *Rng) Vec {
        while (true) {
            const p: Vec = .{ rng.range(-1, 1), rng.range(-1, 1), 0 };
            if (dot(p, p) < 1) return p;
        }
    }
};

const Ray = struct {
    origin: Vec,
    direction: Vec,

    fn at(ray: Ray, t: f32) Vec {
        return ray.origin + splat(t) * ray.direction;
    }

    fn hitSphere(ray: Ray, sphere: scene.Sphere, t_min: f32, t_max: f32) ?Hit {
        const center = sphere.center.vector();
        const oc = center - ray.origin;
        const a = dot(ray.direction, ray.direction);
        const h = dot(ray.direction, oc);
        const c = dot(oc, oc) - sphere.radius * sphere.radius;
        const discriminant = h * h - a * c;
        if (discriminant < 0) return null;
        const root = @sqrt(discriminant);
        const t = for ([2]f32{ (h - root) / a, (h + root) / a }) |t| {
            if (t_min < t and t < t_max) break t;
        } else return null;
        const point = ray.at(t);
        const outward_normal = (point - center) / splat(sphere.radius);
        const front_face = dot(ray.direction, outward_normal) < 0;
        return .{
            .point = point,
            .normal = if (front_face) outward_normal else -outward_normal,
            .t = t,
            .front_face = front_face,
            .material = sphere.material,
        };
    }

    fn hitWorld(ray: Ray, world: []addrspace(.global) const scene.Sphere) ?Hit {
        var closest: ?Hit = null;
        var t_max = std.math.inf(f32);
        for (world) |sphere| {
            if (ray.hitSphere(sphere, 0.001, t_max)) |hit| {
                closest = hit;
                t_max = hit.t;
            }
        }
        return closest;
    }

    fn color(
        ray: Ray,
        world: []addrspace(.global) const scene.Sphere,
        max_depth: u32,
        rng: *Rng,
    ) Vec {
        var current = ray;
        var attenuation = splat(1);
        for (0..max_depth) |_| {
            const hit = current.hitWorld(world) orelse {
                const a = 0.5 * (unit(current.direction)[1] + 1);
                return attenuation * (splat(1 - a) + splat(a) * Vec{ 0.5, 0.7, 1 });
            };
            const scattered = hit.scatter(current, rng) orelse return splat(0);
            attenuation *= scattered.attenuation;
            current = scattered.ray;
        }
        return splat(0);
    }
};

const Hit = struct {
    point: Vec,
    normal: Vec,
    t: f32,
    front_face: bool,
    material: scene.Material,

    fn scatter(hit: Hit, ray: Ray, rng: *Rng) ?Scatter {
        const payload = hit.material.payload;
        switch (hit.material.tag) {
            .lambertian => {
                var direction = hit.normal + rng.unitVector();
                if (@reduce(.And, @abs(direction) < splat(1e-8))) direction = hit.normal;
                return .{
                    .attenuation = payload.lambertian.albedo.vector(),
                    .ray = .{ .origin = hit.point, .direction = direction },
                };
            },
            .metal => {
                const metal = payload.metal;
                const fuzz = splat(metal.fuzz) * rng.unitVector();
                const reflected = unit(reflect(ray.direction, hit.normal)) + fuzz;
                if (dot(reflected, hit.normal) <= 0) return null;
                return .{
                    .attenuation = metal.albedo.vector(),
                    .ray = .{ .origin = hit.point, .direction = reflected },
                };
            },
            .dielectric => {
                const refraction_index = payload.dielectric.refraction_index;
                const ri = if (hit.front_face) 1 / refraction_index else refraction_index;
                const unit_direction = unit(ray.direction);
                const cos_theta = @min(dot(-unit_direction, hit.normal), 1);
                const sin_theta = @sqrt(1 - cos_theta * cos_theta);
                const cannot_refract = ri * sin_theta > 1;
                const direction = if (cannot_refract or reflectance(cos_theta, ri) > rng.float())
                    reflect(unit_direction, hit.normal)
                else
                    refract(unit_direction, hit.normal, ri);
                return .{
                    .attenuation = splat(1),
                    .ray = .{ .origin = hit.point, .direction = direction },
                };
            },
        }
    }
};

const Scatter = struct {
    attenuation: Vec,
    ray: Ray,
};

fn reflect(v: Vec, n: Vec) Vec {
    return v - splat(2 * dot(v, n)) * n;
}

fn refract(uv: Vec, n: Vec, etai_over_etat: f32) Vec {
    const cos_theta = @min(dot(-uv, n), 1);
    const perpendicular = splat(etai_over_etat) * (uv + splat(cos_theta) * n);
    const parallel = splat(-@sqrt(@abs(1 - dot(perpendicular, perpendicular)))) * n;
    return perpendicular + parallel;
}

/// Schlick's approximation of the reflectance of a dielectric.
fn reflectance(cosine: f32, refraction_index: f32) f32 {
    const r0 = (1 - refraction_index) / (1 + refraction_index);
    const x = 1 - cosine;
    return r0 * r0 + (1 - r0 * r0) * x * x * x * x * x;
}

fn getRay(camera: scene.Camera, x: u32, y: u32, rng: *Rng) Ray {
    const pixel_x = @as(f32, @floatFromInt(x)) + rng.float() - 0.5;
    const pixel_y = @as(f32, @floatFromInt(y)) + rng.float() - 0.5;
    const sample = camera.pixel00.vector() + splat(pixel_x) * camera.pixel_delta_u.vector() +
        splat(pixel_y) * camera.pixel_delta_v.vector();
    const origin = if (camera.defocus_angle <= 0) camera.center.vector() else origin: {
        const p = rng.inUnitDisk();
        break :origin camera.center.vector() + splat(p[0]) * camera.defocus_disk_u.vector() +
            splat(p[1]) * camera.defocus_disk_v.vector();
    };
    return .{ .origin = origin, .direction = sample - origin };
}

/// Adds `samples` paths of each pixel, numbered from `first_sample`, to `accumulator`.
export fn render(
    accumulator: [*]addrspace(.global) scene.Vec3,
    spheres: [*]addrspace(.global) const scene.Sphere,
    sphere_count: u32,
    camera: scene.Camera,
    max_depth: u32,
    first_sample: u32,
    samples: u32,
) callconv(.kernel) void {
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);
    if (x >= camera.image_width or y >= camera.image_height) return;
    const pixel = y * camera.image_width + x;
    const world = spheres[0..sphere_count];
    // Each pixel and each launch needs its own sequence of random numbers.
    var rng: Rng = .{ .state = pixel *% 0x9e3779b9 +% first_sample };
    var sum = splat(0);
    for (0..samples) |_| sum += getRay(camera, x, y, &rng).color(world, max_depth, &rng);
    accumulator[pixel] = .fromVector(accumulator[pixel].vector() + sum);
}

/// Writes the mean color of each pixel with the gamma 2 of section 9.5.
export fn resolve(
    accumulator: [*]addrspace(.global) const scene.Vec3,
    pixels: [*]addrspace(.global) scene.Rgba8,
    pixel_count: u32,
    samples: u32,
) callconv(.kernel) void {
    const pixel = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    if (pixel >= pixel_count) return;
    const linear = accumulator[pixel].vector() / splat(@floatFromInt(samples));
    const gamma = @sqrt(@max(linear, splat(0)));
    const bytes: @Vector(3, u8) = @intFromFloat(splat(256) * @min(gamma, splat(0.999)));
    pixels[pixel] = .{ .r = bytes[0], .g = bytes[1], .b = bytes[2], .a = 255 };
}
