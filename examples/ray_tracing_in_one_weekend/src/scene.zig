//! The data of the final scene of "Ray Tracing in One Weekend" and the vector functions that the
//! host program and the kernels share. The host program builds the scene and the camera, and the
//! kernels read them, so each type here has an `extern` or `packed` layout.

const std = @import("std");

pub const Vec3 = extern struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn init(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn fromVector(v: @Vector(3, f32)) Vec3 {
        return .{ .x = v[0], .y = v[1], .z = v[2] };
    }

    pub fn vector(v: Vec3) @Vector(3, f32) {
        return .{ v.x, v.y, v.z };
    }
};

pub const Material = extern struct {
    tag: Tag,
    payload: Payload,

    pub const Tag = enum(u32) { lambertian, metal, dielectric };

    pub const Lambertian = extern struct { albedo: Vec3 };
    pub const Metal = extern struct { albedo: Vec3, fuzz: f32 };
    pub const Dielectric = extern struct { refraction_index: f32 };

    pub const Payload = extern union {
        lambertian: Lambertian,
        metal: Metal,
        dielectric: Dielectric,
    };

    pub fn lambertian(albedo: Vec3) Material {
        return .{ .tag = .lambertian, .payload = .{ .lambertian = .{ .albedo = albedo } } };
    }

    pub fn metal(albedo: Vec3, fuzz: f32) Material {
        return .{ .tag = .metal, .payload = .{ .metal = .{ .albedo = albedo, .fuzz = fuzz } } };
    }

    pub fn dielectric(refraction_index: f32) Material {
        const payload: Payload = .{ .dielectric = .{ .refraction_index = refraction_index } };
        return .{ .tag = .dielectric, .payload = payload };
    }
};

pub const Sphere = extern struct {
    center: Vec3,
    radius: f32,
    material: Material,
};

pub const Camera = extern struct {
    image_width: u32,
    image_height: u32,
    center: Vec3,
    pixel00: Vec3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    defocus_angle: f32,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,

    pub const Options = struct {
        aspect_ratio: f32,
        image_width: u32,
        /// The vertical field of view in degrees.
        vfov: f32,
        lookfrom: Vec3,
        lookat: Vec3,
        vup: Vec3,
        /// The angle in degrees of the cone with its apex at the viewport center and its base at
        /// the camera center.
        defocus_angle: f32,
        focus_dist: f32,
    };

    /// Computes the viewport and the defocus disk as the book's `camera::initialize()` does.
    pub fn init(options: Options) Camera {
        const width: f32 = @floatFromInt(options.image_width);
        const image_height: u32 = @max(1, @as(u32, @intFromFloat(width / options.aspect_ratio)));
        const height: f32 = @floatFromInt(image_height);
        const center = options.lookfrom.vector();
        const h = @tan(std.math.degreesToRadians(options.vfov) / 2);
        const viewport_height = 2 * h * options.focus_dist;
        const viewport_width = viewport_height * (width / height);
        const w = unit(center - options.lookat.vector());
        const u = unit(cross(options.vup.vector(), w));
        const v = cross(w, u);
        const viewport_u = splat(viewport_width) * u;
        const viewport_v = splat(-viewport_height) * v;
        const pixel_delta_u = viewport_u / splat(width);
        const pixel_delta_v = viewport_v / splat(height);
        const focus = splat(options.focus_dist) * w;
        const upper_left = center - focus - viewport_u / splat(2) - viewport_v / splat(2);
        const defocus_angle = std.math.degreesToRadians(options.defocus_angle);
        const defocus_radius = options.focus_dist * @tan(defocus_angle / 2);
        return .{
            .image_width = options.image_width,
            .image_height = image_height,
            .center = .fromVector(center),
            .pixel00 = .fromVector(upper_left + splat(0.5) * (pixel_delta_u + pixel_delta_v)),
            .pixel_delta_u = .fromVector(pixel_delta_u),
            .pixel_delta_v = .fromVector(pixel_delta_v),
            .defocus_angle = options.defocus_angle,
            .defocus_disk_u = .fromVector(splat(defocus_radius) * u),
            .defocus_disk_v = .fromVector(splat(defocus_radius) * v),
        };
    }
};

pub const Rgba8 = packed struct(u32) {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

pub fn splat(x: f32) @Vector(3, f32) {
    return @splat(x);
}

pub fn dot(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return @reduce(.Add, a * b);
}

pub fn cross(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

pub fn unit(v: @Vector(3, f32)) @Vector(3, f32) {
    return v / splat(@sqrt(dot(v, v)));
}
