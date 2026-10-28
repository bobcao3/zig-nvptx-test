#!/usr/bin/env bash
set -euo pipefail

bootstrap_version='0.18.0-dev.92+1e1964473'
linux_dist='zig-nvptx-x86_64-linux-musl-x86_64_v3'
macos_dist='zig-nvptx-aarch64-macos-none-apple_m1'

checkout_compiler() {
    git init compiler-source
    git -C compiler-source remote add origin https://forge.bc3.moe/bob/zig-nvptx.git
    git -C compiler-source fetch --depth=1 origin "$SOURCE_REVISION"
    git -C compiler-source checkout --detach FETCH_HEAD
    test "$(git -C compiler-source rev-parse HEAD)" = "$SOURCE_REVISION"
}

download_toolchain() {
    local target=$1 checksum=$2
    local name="zig+llvm+lld+clang-$target-$bootstrap_version"
    curl --fail --location --retry 3 --output "$RUNNER_TEMP/$name.tar.xz" \
        "https://ziglang.org/deps/$name.tar.xz"
    echo "$checksum  $RUNNER_TEMP/$name.tar.xz" | shasum -a 256 -c -
    tar -xf "$RUNNER_TEMP/$name.tar.xz" -C "$RUNNER_TEMP"
    rm "$RUNNER_TEMP/$name.tar.xz"
}

package_compiler() {
    local dist=$1
    cp LICENSE README.md "$RUNNER_TEMP/$dist/"
    test -f "$RUNNER_TEMP/$dist/lib/std/std.zig"
    tar -C "$RUNNER_TEMP" -cJf "$GITHUB_WORKSPACE/$dist.tar.xz" "$dist"
}

source_version() {
    python3 -c 'import re,sys; s=open("build.zig").read(); m=re.search(r"const zig_version: std.SemanticVersion = .\{ .major = (\d+), .minor = (\d+), .patch = (\d+) \};", s); assert m; print(".".join(m.groups()) + "-dev.0+" + sys.argv[1][:9])' "$SOURCE_REVISION"
}

build_linux() {
    local toolchain="$RUNNER_TEMP/zig+llvm+lld+clang-x86_64-linux-musl-$bootstrap_version"
    local version
    cd compiler-source
    version=$(source_version)
    export ZIG_GLOBAL_CACHE_DIR="$RUNNER_TEMP/zig-global-cache"
    export ZIG_LOCAL_CACHE_DIR="$RUNNER_TEMP/zig-local-cache"
    "$toolchain/bin/zig" build --zig-lib="$PWD/lib" \
        --prefix "$RUNNER_TEMP/$linux_dist" --search-prefix "$toolchain" --maxrss 24000000000 \
        -Dtarget=x86_64-linux-musl -Dcpu=x86_64_v3 -Doptimize=ReleaseFast \
        -Dstatic-llvm -Duse-zig-libcxx -Dflat -Dno-langref -Dstrip \
        -Dversion-string="$version" -j2
    package_compiler "$linux_dist"
}

build_macos() {
    local toolchain="$RUNNER_TEMP/zig+llvm+lld+clang-aarch64-macos-none-$bootstrap_version"
    local linux_toolchain="$RUNNER_TEMP/zig+llvm+lld+clang-x86_64-linux-musl-$bootstrap_version"
    local zig="$RUNNER_TEMP/host-compiler/bin/zig"
    local version
    cd compiler-source
    version=$(source_version)
    export ZIG_GLOBAL_CACHE_DIR="$RUNNER_TEMP/host-global-cache"
    export ZIG_LOCAL_CACHE_DIR="$RUNNER_TEMP/host-local-cache"
    "$linux_toolchain/bin/zig" build --zig-lib="$PWD/lib" \
        --prefix "$RUNNER_TEMP/host-compiler" --search-prefix "$linux_toolchain" --maxrss 24000000000 \
        -Dtarget=x86_64-linux-musl -Dcpu=x86_64_v3 -Doptimize=ReleaseFast \
        -Dstatic-llvm -Duse-zig-libcxx -Dno-lib -Dstrip -Dversion-string="$version" -j2
    rm -rf "$RUNNER_TEMP/host-local-cache" "$RUNNER_TEMP/host-global-cache" "$linux_toolchain"
    export ZIG_GLOBAL_CACHE_DIR="$RUNNER_TEMP/zig-global-cache"
    export ZIG_LOCAL_CACHE_DIR="$RUNNER_TEMP/zig-local-cache"
    "$zig" build --zig-lib="$PWD/lib" \
        --prefix "$RUNNER_TEMP/$macos_dist" --search-prefix "$toolchain" --maxrss 24000000000 \
        -Dtarget=aarch64-macos-none -Dcpu=apple_m1 -Doptimize=ReleaseFast \
        -Dstatic-llvm -Duse-zig-libcxx -Dflat -Dno-langref -Dstrip \
        -Dversion-string="$("$zig" version)" -j2
    package_compiler "$macos_dist"
}

verify_package() {
    local dist=$1 archive=$2
    mkdir -p "$RUNNER_TEMP/package-check"
    tar -xf "$archive" -C "$RUNNER_TEMP/package-check"
    local zig="$RUNNER_TEMP/package-check/$dist/zig"
    unset ZIG_LIB_DIR
    "$zig" version
    printf '%s\n' 'const std = @import("std");' \
        'pub fn main() void { std.debug.assert(std.mem.eql(u8, "zig", "zig")); }' \
        > "$RUNNER_TEMP/package-check/host.zig"
    "$zig" build-exe "$RUNNER_TEMP/package-check/host.zig" -fllvm \
        -femit-bin="$RUNNER_TEMP/package-check/host"
    "$RUNNER_TEMP/package-check/host"
    printf '%s\n' 'export fn kernel(out: *u32) callconv(.kernel) void { out.* = 42; }' \
        > "$RUNNER_TEMP/package-check/kernel.zig"
    "$zig" build-exe "$RUNNER_TEMP/package-check/kernel.zig" -target nvptx64-cuda \
        -mcpu=sm_80 -femit-bin="$RUNNER_TEMP/package-check/kernel.ptx"
    grep -q '\.entry kernel' "$RUNNER_TEMP/package-check/kernel.ptx"
}

publish_release() {
    local linux="$linux_dist.tar.xz" macos="$macos_dist.tar.xz"
    test -s "packages/$linux"
    test -s "packages/$macos"
    (cd packages && sha256sum "$linux" "$macos" > SHA256SUMS)

    test -n "$FORGEJO_RELEASE_TOKEN"
    local timestamp tag
    timestamp=$(date -u '+%Y-%m-%d-%H%M%SZ')
    tag="nvptx-$timestamp"
    local payload response
    payload=$(jq -n --arg tag "$tag" --arg target "$GITHUB_SHA" '{tag_name: $tag, target: $target}')
    response=$(curl --fail-with-body --silent --show-error \
        -H "Authorization: token $FORGEJO_RELEASE_TOKEN" -H 'Content-Type: application/json' \
        --data "$payload" https://forge.bc3.moe/api/v1/repos/bob/zig-nvptx-test/tags)
    test "$(jq -r '.commit.sha' <<< "$response")" = "$GITHUB_SHA"

    local mirrored_revision=''
    for ((attempt = 0; attempt < 60; attempt++)); do
        mirrored_revision=$(git ls-remote "https://github.com/$GH_REPO.git" "refs/tags/$tag" | cut -f1) || true
        if [[ "$mirrored_revision" == "$GITHUB_SHA" ]]; then
            break
        fi
        sleep 2
    done
    test "$mirrored_revision" = "$GITHUB_SHA"

    gh release create "$tag" "packages/$linux" "packages/$macos" packages/SHA256SUMS \
        --verify-tag --prerelease --title "Zig NVPTX $timestamp" \
        --notes "Compiler source: https://forge.bc3.moe/bob/zig-nvptx/commit/$SOURCE_REVISION
Bootstrap: Zig $bootstrap_version with LLVM, Clang and LLD."
}

case "${1:-}" in
    source)
        revision=$(git ls-remote https://forge.bc3.moe/bob/zig-nvptx.git refs/heads/nvptx | cut -f1)
        [[ "$revision" =~ ^[0-9a-f]{40}$ ]]
        echo "revision=$revision" >> "$GITHUB_OUTPUT"
        ;;
    linux-download)
        checkout_compiler
        download_toolchain x86_64-linux-musl 3d59a8b9ac5ed3c1a046e550f35b5fb89cb5cd5ebf177cdf1ca00ad8150994e1
        ;;
    linux-build) build_linux ;;
    linux-check) verify_package "$linux_dist" "$GITHUB_WORKSPACE/$linux_dist.tar.xz" ;;
    macos-download)
        checkout_compiler
        download_toolchain x86_64-linux-musl 3d59a8b9ac5ed3c1a046e550f35b5fb89cb5cd5ebf177cdf1ca00ad8150994e1
        download_toolchain aarch64-macos-none 5daddeaad6e135ae4bd09d6c2b4c5a30873b58d10671c13fb8ac1af246df05e6
        ;;
    macos-build) build_macos ;;
    macos-check) verify_package "$macos_dist" "$GITHUB_WORKSPACE/packages/$macos_dist.tar.xz" ;;
    publish) publish_release ;;
    *) echo "usage: $0 {source|linux-download|linux-build|linux-check|macos-download|macos-build|macos-check|publish}" >&2; exit 2 ;;
esac
