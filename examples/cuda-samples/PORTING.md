# Porting a sample

`0_Introduction/vectorAdd` and `0_Introduction/matrixMul` are the templates. The upstream sources
are in `.upstream/cpp/<category>/<sample>/`, which is not under version control:

    git clone --depth 1 https://github.com/NVIDIA/cuda-samples .upstream

`zig build --fetch` also fetches the upstream commit that `build.zig.zon` pins into `zig-pkg/`. The
build embeds the input files of the samples from that copy, and the `compare-nvcc` step compiles
its `.cu` files.

## Files of a sample

`build.zig` finds every folder `<category>/<sample>/` that contains `<sample>.zig`; no build entry
is needed. In that folder:

| File | Role |
|---|---|
| `<sample>.zig` | The host program, with `pub fn main`. It imports the module `cuda`. |
| `<name>_kernel.zig` | One PTX module per file. It uses `std.nvptx` for PTX instructions. The host gets the PTX text as `@embedFile("<name>_kernel.ptx")`. |
| other `.zig` files | Code that the host program or a kernel file imports by relative path. |

The host program reads each input file of the original sample with `@embedFile("<file name>")`.
`build.zig` provides every file of the `data` folder of the upstream sample under its own name, so
the input files stay out of this repository.

Keep the upstream names: the sample folder, the file stems (`matrixMul_kernel.cu` becomes
`matrixMul_kernel.zig`; a kernel in `<sample>.cu` goes to `<sample>_kernel.zig`), the kernel and
function names converted to Zig's case rules (`MatrixMulCUDA` becomes `matrixMulCuda`,
`vectorLength` becomes `vector_length`), and the structure and order of the original code, so that
a reader can compare the two side by side.

A kernel keeps the logic and control flow of the original: the same memory accesses, barriers,
loops and branches, in the same order. Write that logic in idiomatic Zig rather than in C++ style:

- `comptime` parameters and `inline for` replace templates, `#define` constants and `#pragma unroll`.
- `switch` replaces `if`/`else` chains over one value, and tagged unions or enums replace integer
  mode flags.
- Error unions and `try` replace `checkCudaErrors`, and `defer` frees each resource next to the
  code that creates it.
- Slices, `for` over several sequences, and `@Vector` replace pointer arithmetic where the PTX stays
  the same.

Every ported file starts with the copyright notice of the upstream file that it ports, with the same
year, turned into `//` comments (the text in `0_Introduction/vectorAdd/vectorAdd.zig`). The upstream
description of the sample follows as a `//!` comment. The license notice does not count toward the
comment budget.

## Host program

    const std = @import("std");
    const cuda = @import("cuda");

    pub fn main(init: std.process.Init) !void {
        const args = try init.minimal.args.toSlice(init.arena.allocator());
        const device = try cuda.init(0);
        defer device.deinit();
        const module = try cuda.Module.load(@embedFile("vectorAdd_kernel.ptx"));
        defer module.unload();
        const vec_add = try module.function("vecAdd");
        const a = try cuda.dupe(f32, host_a);            // cudaMalloc + cudaMemcpy
        defer a.free();
        try vec_add.launch(.{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, .{ a, b, c, @as(i32, n) });
        try c.copyTo(host_c);                            // cudaMemcpy, waits for the kernel
    }

| CUDA runtime | `../../cuda.zig` |
|---|---|
| `cudaSetDevice`, `findCudaDevice` | `cuda.init(ordinal)`, `device.deinit()` |
| `cudaGetDeviceProperties` | `device.computeCapability()`, `device.attribute(.multiprocessor_count)`, `device.name(&buffer)` |
| `cudaMalloc`, `cudaFree` | `cuda.alloc(T, len)`, `buffer.free()` |
| `cudaMallocManaged` | `cuda.allocManaged(T, len)`, `buffer.hostSlice()` |
| `cudaMallocHost` | an ordinary allocation from `init.gpa` |
| `cudaMemcpy`, `cudaMemcpyAsync` | `buffer.copyFrom(slice)`, `copyTo`, `copyFromBuffer`, `copyFromAsync(slice, stream)`, `copyToAsync` |
| `cudaMemset` | `buffer.memset(byte)` |
| `cudaMemcpyToSymbol`, `cudaGetSymbolAddress` | `module.global(T, "name")`, then `copyFrom` |
| `kernel<<<grid, block, shared, stream>>>(args)` | `function.launch(.{ .grid, .block, .shared_memory_bytes, .stream }, .{args})` |
| `cudaLaunchCooperativeKernel` | `.cooperative = true` in the launch config |
| `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` | `function.setMaxDynamicSharedMemory(bytes)` |
| `cudaDeviceSynchronize` | `cuda.synchronize()` |
| `cudaStreamCreate`, `cudaStreamSynchronize` | `cuda.Stream.create()`, `stream.synchronize()`, `stream.destroy()` |
| `cudaEventCreate`, `Record`, `Synchronize`, `ElapsedTime` | `cuda.Event.create()`, `event.record(stream)`, `event.synchronize()`, `start.elapsed(stop)` |
| `checkCudaErrors(...)` | `try`; a failing driver call logs the driver's message and returns a Zig error |

A launch argument is a `Buffer` (the kernel gets its device address) or a value of the kernel
parameter's type. Cast integer literals: `@as(i32, n)`.

If a sample needs a driver function that `../../cuda.zig` lacks, add the function to `../../cuda.zig`; do not call the
driver from the sample.

Verification:

- Keep the original sizes, iteration counts, tolerances, CPU reference and messages (`Result =
  PASS`, `Test passed`, ...). A fixed seed may replace `srand(time(0))`.
- Exit with a non-zero status when the check fails (`std.process.exit(1)`), even where the original
  only prints the failure: `zig build run` relies on the exit status.
- Print through `std.log`: `std.log.info` for normal output, `std.log.warn` for waivers,
  `std.log.err` for failures and `std.log.debug` for verbose detail. Do not use `std.debug.print`.
- Command line flags of the original (`-wA=320`) keep working; `matrixMul.zig` has a flag parser to
  copy. Flags that select a device or a file are optional.

## Kernels

    const std = @import("std");

    export fn vecAdd(
        a: [*]addrspace(.global) const f32,
        b: [*]addrspace(.global) const f32,
        c: [*]addrspace(.global) f32,
        vector_length: i32,
    ) callconv(.kernel) void {
        const work_index = @workItemId(0) + @workGroupId(0) * @workGroupSize(0);
        if (work_index < vector_length) c[work_index] = a[work_index] + b[work_index];
    }

| CUDA C++ | Zig |
|---|---|
| `__global__ void f(float *p)` | `export fn f(p: [*]addrspace(.global) f32) callconv(.kernel) void` |
| `template <int N> __global__ ...` | a function that returns a struct with the kernel and `@export(&kernel, .{ .name = ... })`; see `matrixMul_kernel.zig` |
| `threadIdx`, `blockIdx`, `blockDim`, `gridDim` | `@workItemId(d)`, `@workGroupId(d)`, `@workGroupSize(d)`, `std.nvptx.nctaid(d)` |
| `__shared__ float s[N][N]` | container-level `var s: [N][N]f32 addrspace(.shared) = undefined;` (inside the struct of a generic kernel) |
| `extern __shared__ float s[]` | `extern var s: [0]f32 align(16) addrspace(.shared);`, then `const p: [*]addrspace(.shared) f32 = &s;` |
| `__constant__ float c[N]` written by the host | `export var c: [N]f32 addrspace(.global) = undefined;` ([issue](issues/constant-memory.md)) |
| `__device__` function | a Zig function; the linker inlines every direct call except calls to `noinline` functions, `@call(.never_inline, ...)` and recursive calls |
| `__syncthreads()` | `std.nvptx.barrierSyncAligned(.work_group, null)` |
| `__syncwarp(mask)` | `std.nvptx.barWarpSync(mask)` |
| `__threadfence()`, `_block()`, `_system()` | `std.nvptx.fence(.seq_cst, .gpu)`, `fence(.seq_cst, .cta)`, `fence(.seq_cst, .sys)` |
| `__shfl_sync(mask, v, lane, width)`, `_up_`, `_down_`, `_xor_` | `std.nvptx.shflSync(.idx, T, v, lane, .fromWidth(.idx, width), mask)`, `.up`, `.down`, `.bfly` (32-bit `T`) |
| `__ballot_sync`, `__all_sync`, `__any_sync`, `__uni_sync`, `__activemask` | `std.nvptx.voteBallotSync(p, mask)`, `voteSync(.all, p, mask)`, `voteSync(.any, ...)`, `voteSync(.uni, ...)`, `activemask()` |
| `__reduce_add_sync` and the other `__reduce_*_sync` (sm_80) | `std.nvptx.reduxSync(.add, T, v, mask)` (`u32` or `i32`) |
| `clock()`, `clock64()`, lane index | `std.nvptx.clock()`, `clock64()`, `laneid()` |
| `atomicAdd(p, v)` and the other atomics | `@atomicRmw(T, p, .Add, v, .monotonic)`; `atomicCAS` is `@cmpxchgStrong(T, p, expected, new, .monotonic, .monotonic)` |
| `atomicAdd_system` | inline PTX `atom.sys...` |
| `sqrtf`, `fabsf`, `floorf`, `fmaf` | `@sqrt`, `@abs`, `@floor`, `@mulAdd` |
| `expf`, `logf`, `sinf`, `cosf`, `powf` | `@exp`, `@log`, `@sin`, `@cos`, `std.math.pow`, which call compiler_rt routines; `build.zig` does not link compiler_rt into the kernels, so a port cannot use them yet |
| `__expf(x)`, `__logf(x)`, `__powf(x, y)` | `std.nvptx.ex2Approx(x * std.math.log2e)`, `lg2Approx(x) * std.math.ln2`, `ex2Approx(lg2Approx(x) * y)`: the instructions that nvcc emits; `std.nvptx.math.ex2ApproxF32(true, x)` and its siblings are the `.ftz` forms of `-use_fast_math` |
| `__sinf`, `__cosf`, `rsqrtf`, `__fdividef` | `std.nvptx.sinApprox`, `cosApprox`, `rsqrtApprox`, `divApprox` |
| `-use_fast_math` on `float` `+`, `-`, `*`, `/` | `@setFloatMode(.optimized)`: the backend emits `.ftz` arithmetic and `div.approx.ftz.f32` |
| other PTX-only operations | inline assembly, as in `lib/std/nvptx.zig` of the compiler |
| `half`, `half2` | `f16`, `@Vector(2, f16)` |
| cooperative groups in a block | `thread_block::sync` is `barrierSyncAligned(.work_group, null)`; a `tiled_partition<32>` is the warp with shuffles and the member mask `0xffffffff`; compute `thread_rank` from the work item IDs |
| WMMA (`wmma::load_matrix_sync`, `mma_sync`, ...) | the PTX `wmma.load`, `wmma.mma` and `wmma.store` instructions, which nvcc emits for WMMA, in inline assembly with `@Vector` operands |
| `cp.async`, `memcpy_async`, `cuda::pipeline` | inline PTX `cp.async...`, `cp.async.commit_group`, `cp.async.wait_group` |
| `printf` in a kernel | `extern fn vprintf(format: [*:0]const u8, args: ?*const anyopaque) callconv(.c) i32;` with an `extern struct` of the arguments (runs; the driver prints after the next synchronization) |
| `assert` in a kernel | `std.debug.assert` traps in safe and debug builds; the host gets `error.LaunchFailed` without a message |

Inline assembly: outputs `"=h"` (16-bit), `"=r"` (32-bit), `"=l"` (64-bit), `"=f"` (`f32`), `"=d"`
(`f64`); `"+f"`
for read-write; a `@Vector(n, T)` operand becomes a register list `{...}`; `"n"` takes a
comptime-known immediate; a `.shared` pointer for an `"r"` operand passes its 32-bit shared address.
Add `.{ .memory = true }` as clobber for instructions that touch memory. Examples:
`zig-nvptx-test/suites/kernels/{tma,gemm,wgmma}.zig`.

`f16` and `@Vector(2, f16)` arithmetic, `f32` atomic add on shared memory,
`u64` atomics, `@cmpxchgStrong`, device recursion, `mma.sync` with vector operands, `cp.async`,
`@panic` and kernel parameters compile and pass ptxas for `sm_80`. Kernel parameters over 4352
bytes in total need PTX ISA 8.1, and the backend raises `.version` to 8.1 for them.

## Build, run, check

    $ZIG_NVPTX_PREFIX/bin/zig build build-<sample>   # zig-out/bin/<category>/<sample>/<sample> and *.ptx
    $ZIG_NVPTX_PREFIX/bin/zig build run-<sample>     # -Dgpu=sm_80+ptx78 -Dkernel-optimize=fast
    $ZIG_NVPTX_PREFIX/bin/zig build run-<sample> -Dgpu=sm_90a -Dkernel-optimize=safe
    ptxas -arch=sm_80 zig-out/bin/<category>/<sample>/<name>_kernel.ptx -o /dev/null

Use the per-sample steps while a port is in progress. `nvcc -ptx` of the upstream `.cu` file shows
the PTX that the original produces; it is a reference only, never part of the build.

A port is correct when it behaves as the upstream `.cu` sample does: the same inputs give the same
output and pass the same checks.

A sample is done when `zig fmt --check` passes on its files, `run-<sample>` passes with the
defaults and with `-Dgpu=<the exact target of the GPU> -Dkernel-optimize=safe`, and the PTX passes
ptxas.

## Backend issues

For each compiler problem, write `issues/<sample>-<topic>.md`, or extend the existing issue file
when the problem is the same, and add a row to `BACKEND-ISSUES.md`. The issue file names the
samples that the problem affects, a minimal kernel that shows it, the command and its output, and
the workaround. Use a workaround only if the
port stays faithful to the original; otherwise mark the sample blocked.

## Blocked samples

When a sample cannot be ported because the compiler, `std.nvptx` or `../../cuda.zig`
lacks a feature, its folder holds only `README.md`, with exactly this text:

    TODO

    Upstream: <https://github.com/NVIDIA/cuda-samples/tree/master/cpp/<category>/<sample>>

Remove every other file of the folder. Record the missing feature in `SAMPLES.md`.
