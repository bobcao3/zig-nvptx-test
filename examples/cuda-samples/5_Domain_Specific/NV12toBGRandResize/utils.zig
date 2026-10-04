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

//! Dumps raw BGR frames to files.

const std = @import("std");
const cuda = @import("cuda");

pub fn dumpBgr(
    gpa: std.mem.Allocator,
    io: std.Io,
    d_src_bgr: cuda.Buffer(f32),
    pitch: u32,
    width: u32,
    height: u32,
    batch_size: u32,
    folder: []const u8,
    tag: []const u8,
) !void {
    var directory_buffer: [120]u8 = undefined;
    const directory = try std.fmt.bufPrint(&directory_buffer, "output/{s}", .{folder});
    try std.Io.Dir.cwd().createDirPath(io, directory);

    const frame_len = width * height * 3;
    const bgr = try gpa.alloc(f32, frame_len);
    defer gpa.free(bgr);

    for (0..batch_size) |i| {
        const frame_offset = i * pitch * height * 3;
        const d_bgr: cuda.Buffer(f32) = .{
            .address = d_src_bgr.address + frame_offset * @sizeOf(f32),
            .len = frame_len,
        };
        try d_bgr.copyTo(bgr);

        var filename_buffer: [256]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buffer, "{s}/{s}_{d}.raw", .{
            directory,
            tag,
            i + 1,
        });
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = filename,
            .data = std.mem.sliceAsBytes(bgr),
        });
    }
}
