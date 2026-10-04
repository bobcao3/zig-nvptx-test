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

//! The header of a DDS file.

pub fn makeFourcc(comptime chars: *const [4]u8) u32 {
    return @as(u32, chars[0]) | (@as(u32, chars[1]) << 8) | (@as(u32, chars[2]) << 16) | (@as(u32, chars[3]) << 24);
}

pub const PixelFormat = extern struct {
    size: u32,
    flags: u32,
    fourcc: u32,
    bitcount: u32,
    rmask: u32,
    gmask: u32,
    bmask: u32,
    amask: u32,
};

pub const Caps = extern struct {
    caps1: u32,
    caps2: u32,
    caps3: u32,
    caps4: u32,
};

/// The DDS file header.
pub const Header = extern struct {
    fourcc: u32,
    size: u32,
    flags: u32,
    height: u32,
    width: u32,
    pitch: u32,
    depth: u32,
    mipmapcount: u32,
    reserved: [11]u32,
    pf: PixelFormat,
    caps: Caps,
    notused: u32,
};

pub const fourcc_dds = makeFourcc("DDS ");
pub const fourcc_dxt1 = makeFourcc("DXT1");
pub const ddsd_width: u32 = 0x00000004;
pub const ddsd_height: u32 = 0x00000002;
pub const ddsd_caps: u32 = 0x00000001;
pub const ddsd_pixelformat: u32 = 0x00001000;
pub const ddscaps_texture: u32 = 0x00001000;
pub const ddpf_fourcc: u32 = 0x00000004;
pub const ddsd_linearsize: u32 = 0x00080000;
