// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! High quality DXT compression using CUDA. This sample shows how to implement an existing
//! computationally intensive CPU compression algorithm in parallel on the GPU, and obtain an
//! order of magnitude performance improvement.

const std = @import("std");
const cuda = @import("cuda");
const dds = @import("dds.zig");
const permutations_file = @import("permutations.zig");

const input_image = "teapot512_std.ppm";
const reference_image = "teapot512_ref.dds";

const error_threshold = 0.02;

const num_threads = 64; // The number of work items per work group.

// Helper types and functions to validate the output of the compressor. A bitwise comparison does
// not work, because different compilers produce different results for different targets due to
// floating point arithmetic.

const Color32 = packed struct(u32) { b: u8, g: u8, r: u8, a: u8 };

const Color16 = packed struct(u16) { b: u5, g: u6, r: u5 };

const BlockDxt1 = extern struct {
    col0: Color16,
    col1: Color16,
    indices: u32,

    fn decompress(block: BlockDxt1) [16]Color32 {
        var palette: [4]Color32 = undefined;

        // Does the bit expansion before the interpolation.
        palette[0].b = (@as(u8, block.col0.b) << 3) | (block.col0.b >> 2);
        palette[0].g = (@as(u8, block.col0.g) << 2) | (block.col0.g >> 4);
        palette[0].r = (@as(u8, block.col0.r) << 3) | (block.col0.r >> 2);
        palette[0].a = 0xFF;

        palette[1].r = (@as(u8, block.col1.r) << 3) | (block.col1.r >> 2);
        palette[1].g = (@as(u8, block.col1.g) << 2) | (block.col1.g >> 4);
        palette[1].b = (@as(u8, block.col1.b) << 3) | (block.col1.b >> 2);
        palette[1].a = 0xFF;

        if (@as(u16, @bitCast(block.col0)) > @as(u16, @bitCast(block.col1))) {
            // A four-color block: derive the other two colors.
            palette[2].r = @intCast((2 * @as(u32, palette[0].r) + palette[1].r) / 3);
            palette[2].g = @intCast((2 * @as(u32, palette[0].g) + palette[1].g) / 3);
            palette[2].b = @intCast((2 * @as(u32, palette[0].b) + palette[1].b) / 3);
            palette[2].a = 0xFF;

            palette[3].r = @intCast((2 * @as(u32, palette[1].r) + palette[0].r) / 3);
            palette[3].g = @intCast((2 * @as(u32, palette[1].g) + palette[0].g) / 3);
            palette[3].b = @intCast((2 * @as(u32, palette[1].b) + palette[0].b) / 3);
            palette[3].a = 0xFF;
        } else {
            // A three-color block: derive the other color.
            palette[2].r = @intCast((@as(u32, palette[0].r) + palette[1].r) / 2);
            palette[2].g = @intCast((@as(u32, palette[0].g) + palette[1].g) / 2);
            palette[2].b = @intCast((@as(u32, palette[0].b) + palette[1].b) / 2);
            palette[2].a = 0xFF;

            palette[3] = .{ .r = 0x00, .g = 0x00, .b = 0x00, .a = 0x00 };
        }

        var colors: [16]Color32 = undefined;
        for (&colors, 0..) |*color, i| {
            color.* = palette[(block.indices >> @intCast(2 * i)) & 0x3];
        }
        return colors;
    }
};

fn compareColors(b0: *const [16]Color32, b1: *const [16]Color32) i32 {
    var sum: i32 = 0;

    for (b0, b1) |c0, c1| {
        const r = @as(i32, c0.r) - c1.r;
        const g = @as(i32, c0.g) - c1.g;
        const b = @as(i32, c0.b) - c1.b;
        sum += r * r + g * g + b * b;
    }

    return sum;
}

fn compareBlock(b0: BlockDxt1, b1: BlockDxt1) i32 {
    if (std.mem.eql(u8, std.mem.asBytes(&b0), std.mem.asBytes(&b1))) {
        return 0;
    } else {
        return compareColors(&b0.decompress(), &b1.decompress());
    }
}

/// An image of 4 bytes for each pixel.
const Image = struct {
    width: u32,
    height: u32,
    data: []u8,
};

/// Loads a binary PPM file and converts the pixels to 4 bytes with zero alpha.
fn loadPpm4ub(gpa: std.mem.Allocator, file: []const u8) !Image {
    var tokens = std.mem.tokenizeAny(u8, file, " \t\r\n");
    if (!std.mem.eql(u8, tokens.next() orelse return error.InvalidImage, "P6")) return error.InvalidImage;

    var header: [3]u32 = undefined;
    for (&header) |*field| {
        var token = tokens.next() orelse return error.InvalidImage;
        // Skip the comment lines.
        while (token[0] == '#') {
            const comment_end = std.mem.indexOfScalarPos(u8, file, tokens.index, '\n') orelse return error.InvalidImage;
            tokens.index = comment_end;
            token = tokens.next() orelse return error.InvalidImage;
        }
        field.* = try std.fmt.parseInt(u32, token, 10);
    }
    const width, const height, const max_value = header;
    if (max_value != 255) return error.InvalidImage;

    // One white space character separates the header from the pixels.
    const pixels = file[tokens.index + 1 ..];
    if (pixels.len < width * height * 3) return error.InvalidImage;

    const data = try gpa.alloc(u8, width * height * 4);
    for (0..width * height) |i| {
        @memcpy(data[i * 4 ..][0..3], pixels[i * 3 ..][0..3]);
        data[i * 4 + 3] = 0;
    }
    return .{ .width = width, .height = height, .data = data };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    std.log.info("{s} Starting...", .{args[0]});

    const device = try cuda.init(0);
    defer device.deinit();
    const module = try cuda.Module.load(@embedFile("dxtc_kernel.ptx"));
    defer module.unload();
    const compress = try module.function("compress");

    // Load the input image.
    const image = try loadPpm4ub(gpa, @embedFile(input_image));
    defer gpa.free(image.data);
    const w = image.width;
    const h = image.height;
    std.log.info("Image Loaded '{s}', {d} x {d} pixels", .{ input_image, w, h });

    // Allocate the input image.
    const mem_size = w * h * 4;
    std.debug.assert(mem_size != 0);
    const block_image = try gpa.alloc(u32, w * h);
    defer gpa.free(block_image);

    // Convert the linear image to a block linear image.
    const pixels = std.mem.bytesAsSlice(u32, image.data);
    for (0..h / 4) |by| {
        for (0..w / 4) |bx| {
            for (0..16) |i| {
                const x = i & 3;
                const y = i / 4;
                block_image[(by * w / 4 + bx) * 16 + i] = pixels[(by * 4 + y) * 4 * (w / 4) + bx * 4 + x];
            }
        }
    }

    // Copy into global memory.
    const d_data = try cuda.alloc(u32, w * h);
    defer d_data.free();

    // The result.
    const compressed_size = (w / 4) * (h / 4) * 8;
    const d_result = try cuda.alloc(u32, compressed_size / @sizeOf(u32));
    defer d_result.free();
    const h_result = try gpa.alloc(u32, compressed_size / @sizeOf(u32));
    defer gpa.free(h_result);

    // Compute the permutations.
    var permutations: [1024]u32 = undefined;
    permutations_file.computePermutations(&permutations);

    // Copy the permutations from the host to the device.
    const d_permutations = try cuda.dupe(u32, &permutations);
    defer d_permutations.free();

    // Copy the image from the host to the device.
    try d_data.copyFrom(block_image);

    // Determine the launch configuration and run the timed computation `num_iterations` times.
    const blocks = ((w + 3) / 4) * ((h + 3) / 4); // This rounds up by 1 block in each dimension if w % 4 != 0.

    // Restrict the number of work groups to launch on low end GPUs to avoid a kernel timeout.
    const multiprocessor_count: u32 = @intCast(try device.attribute(.multiprocessor_count));
    const blocks_per_launch = @min(blocks, 768 * multiprocessor_count);

    std.log.info("Running DXT Compression on {d} x {d} image...", .{ w, h });
    std.log.info("{d} Blocks, {d} Threads per Block, {d} Threads in Grid...", .{
        blocks, num_threads, blocks * num_threads,
    });
    const num_iterations = 1;

    var start: std.Io.Timestamp = undefined;
    var i: i32 = -1;
    while (i < num_iterations) : (i += 1) {
        if (i == 0) {
            try cuda.synchronize();
            start = std.Io.Clock.awake.now(init.io);
        }

        var j: u32 = 0;
        while (j < blocks) : (j += blocks_per_launch) {
            try compress.launch(
                .{ .grid = .{ .x = @min(blocks_per_launch, blocks - j) }, .block = .{ .x = num_threads } },
                .{ d_permutations, d_data, d_result, @as(i32, @intCast(j)) },
            );
        }
    }

    // Sync to the host, stop the timer and record the performance.
    try cuda.synchronize();
    const stop = std.Io.Clock.awake.now(init.io);
    const seconds = @as(f64, @floatFromInt(start.durationTo(stop).toNanoseconds())) * 1.0e-9;
    const avg_time = seconds / num_iterations;
    std.log.info(
        "dxtc, Throughput = {d:.4} MPixels/s, Time = {d:.5} s, Size = {d} Pixels, NumDevsUsed = {d}, Workgroup = {d}",
        .{ 1.0e-6 * @as(f64, @floatFromInt(w * h)) / avg_time, avg_time, w * h, 1, num_threads },
    );

    // Copy the result data from the device to the host.
    try d_result.copyTo(h_result);

    // Write out the result data to a DDS file.
    const output_filename = "teapot512_std.dds";
    const header: dds.Header = .{
        .fourcc = dds.fourcc_dds,
        .size = 124,
        .flags = dds.ddsd_width | dds.ddsd_height | dds.ddsd_caps | dds.ddsd_pixelformat | dds.ddsd_linearsize,
        .height = h,
        .width = w,
        .pitch = compressed_size,
        .depth = 0,
        .mipmapcount = 0,
        .reserved = @splat(0),
        .pf = .{
            .size = 32,
            .flags = dds.ddpf_fourcc,
            .fourcc = dds.fourcc_dxt1,
            .bitcount = 0,
            .rmask = 0,
            .gmask = 0,
            .bmask = 0,
            .amask = 0,
        },
        .caps = .{ .caps1 = dds.ddscaps_texture, .caps2 = 0, .caps3 = 0, .caps4 = 0 },
        .notused = 0,
    };
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(gpa);
    try output.appendSlice(gpa, std.mem.asBytes(&header));
    try output.appendSlice(gpa, std.mem.sliceAsBytes(h_result));
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = output_filename, .data = output.items });

    // Make sure that the generated image is correct.
    const reference_file = @embedFile(reference_image);
    const reference = reference_file[@sizeOf(dds.Header)..][0 .. (w / 4) * (h / 4) * 8];

    std.log.info("Checking accuracy...", .{});
    var rms: f32 = 0;

    var y: u32 = 0;
    while (y < h) : (y += 4) {
        var x: u32 = 0;
        while (x < w) : (x += 4) {
            const reference_block_idx = (y / 4) * (w / 4) + (x / 4);
            const result_block_idx = (y / 4) * (w / 4) + (x / 4);

            const result_block = std.mem.bytesToValue(BlockDxt1, std.mem.sliceAsBytes(h_result)[result_block_idx * 8 ..][0..8]);
            const reference_block = std.mem.bytesToValue(BlockDxt1, reference[reference_block_idx * 8 ..][0..8]);
            const cmp = compareBlock(result_block, reference_block);

            if (cmp != 0) {
                std.log.warn("Deviation at ({d:>4},{d:>4}):\t{d:.6} rms", .{
                    x / 4, y / 4, @as(f32, @floatFromInt(cmp)) / 16 / 3,
                });
            }

            rms += @floatFromInt(cmp);
        }
    }

    rms /= @floatFromInt(w * h * 3);

    std.log.info("RMS(reference, result) = {d:.6}", .{rms});
    if (rms <= error_threshold) {
        std.log.info("Test passed", .{});
    } else {
        std.log.err("Test failed!", .{});
        std.process.exit(1);
    }
}
