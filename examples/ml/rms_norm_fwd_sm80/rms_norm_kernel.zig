const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const Bf16x2 = nvptx.Bf16x2;

pub const hidden = 4096;

/// Eight bf16 values, the 16 bytes of one vector load or store.
pub const Chunk = @Vector(4, u32);
const chunks_per_row = hidden / 8;
const chunks_per_lane = chunks_per_row / 32;
const whole_warp: nvptx.ShuffleClamp = .fromWidth(.bfly, 32);

/// Computes y = x / sqrt(mean(x * x) + epsilon) * weight with f32 arithmetic. Each warp keeps its
/// row of x in registers between the sum and the scaling, so the kernel reads x from memory once.
pub const RmsNorm = struct {
    pub const name = "rms_norm";
    pub const warps = 2;

    pub fn kernel(
        x: [*]addrspace(.global) const Chunk,
        weight: [*]addrspace(.global) const Chunk,
        y: [*]addrspace(.global) Chunk,
        rows: u32,
        epsilon: f32,
    ) callconv(.kernel) void {
        const lane = @workItemId(0) % 32;
        const row = @workGroupId(0) * warps + @workItemId(0) / 32;
        if (row >= rows) return;
        const x_row = x + row * chunks_per_row + lane;
        const y_row = y + row * chunks_per_row + lane;
        var values: [chunks_per_lane]Chunk = undefined;
        inline for (&values, 0..) |*value, i| value.* = x_row[i * 32];
        var sum: f32 = 0;
        inline for (values) |value| {
            inline for (0..4) |k| {
                const pair = Bf16x2.toF32(@fromBackingInt(value[k]));
                sum += pair[0] * pair[0] + pair[1] * pair[1];
            }
        }
        inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
            sum += nvptx.shflSync(.bfly, f32, sum, offset, whole_warp, .all);
        }
        const scale = nvptx.math.rsqrtApproxF32(false, sum * (1.0 / @as(f32, hidden)) + epsilon);
        inline for (values, 0..) |value, i| {
            const factors = weight[lane + i * 32];
            var result: Chunk = undefined;
            inline for (0..4) |k| {
                const pair = Bf16x2.toF32(@fromBackingInt(value[k]));
                const factor = Bf16x2.toF32(@fromBackingInt(factors[k]));
                const normalized = pair * @as(@Vector(2, f32), @splat(scale)) * factor;
                result[k] = @backingInt(Bf16x2.fromF32(normalized[0], normalized[1]));
            }
            y_row[i * 32] = result;
        }
    }
};

/// Reads x and writes y as `RmsNorm` does. The host program reports speeds as shares of this copy.
pub const Copy = struct {
    pub const name = "copy";
    pub const warps = 1;

    pub fn kernel(
        x: [*]addrspace(.global) const Chunk,
        weight: [*]addrspace(.global) const Chunk,
        y: [*]addrspace(.global) Chunk,
        rows: u32,
        epsilon: f32,
    ) callconv(.kernel) void {
        _ = weight;
        _ = epsilon;
        const lane = @workItemId(0) % 32;
        const row = @workGroupId(0) * warps + @workItemId(0) / 32;
        if (row >= rows) return;
        const x_row = x + row * chunks_per_row + lane;
        const y_row = y + row * chunks_per_row + lane;
        var values: [chunks_per_lane]Chunk = undefined;
        inline for (&values, 0..) |*value, i| value.* = x_row[i * 32];
        inline for (values, 0..) |value, i| y_row[i * 32] = value;
    }
};

pub const kernels = .{ RmsNorm, Copy };

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
    }
}
