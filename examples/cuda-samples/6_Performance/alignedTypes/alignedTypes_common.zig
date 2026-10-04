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

/// The element types of the copy test. The misaligned structures have the alignment of their
/// fields. The aligned structures declare an alignment of 4, 8 or 16 bytes.
pub const Kind = enum {
    uint8,
    uint16,
    rgba8_misaligned,
    la32_misaligned,
    rgb32_misaligned,
    rgba32_misaligned,
    rgba8,
    i32,
    la32,
    rgb32,
    rgba32,
    rgba32_2,

    pub fn Element(comptime kind: Kind) type {
        return switch (kind) {
            .uint8 => u8,
            .uint16 => u16,
            .rgba8_misaligned => extern struct { r: u8, g: u8, b: u8, a: u8 },
            .la32_misaligned => extern struct { l: u32, a: u32 },
            .rgb32_misaligned => extern struct { r: u32, g: u32, b: u32 },
            .rgba32_misaligned => extern struct { r: u32, g: u32, b: u32, a: u32 },
            .rgba8 => extern struct { r: u8 align(4), g: u8, b: u8, a: u8 },
            .i32 => u32,
            .la32 => extern struct { l: u32 align(8), a: u32 },
            .rgb32 => extern struct { r: u32 align(16), g: u32, b: u32 },
            .rgba32 => extern struct { r: u32 align(16), g: u32, b: u32, a: u32 },
            .rgba32_2 => extern struct { c1: Kind.rgba32.Element(), c2: Kind.rgba32.Element() },
        };
    }
};
