# CUDA samples in Zig

Zig ports of [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/cpp). The
self-hosted PTX backend of the Zig compiler builds the kernels. Each host program embeds the PTX
and loads it through the CUDA driver at run time, so the build needs no CUDA toolkit and a run needs
only an NVIDIA driver.

## Build and run

Run from this folder with a Zig compiler that has the PTX backend and can build host programs (a
full build of the fork, `$ZIG_NVPTX_PREFIX/bin/zig`):

    $ZIG_NVPTX_PREFIX/bin/zig build                 # every sample: zig-out/bin/<category>/<sample>/
    $ZIG_NVPTX_PREFIX/bin/zig build run             # runs every sample; fails when one fails
    $ZIG_NVPTX_PREFIX/bin/zig build run-matrixMul   # one sample (build-matrixMul only builds it)
    $ZIG_NVPTX_PREFIX/bin/zig build run-matrixMul -- -wA=640 -hB=640

`zig build --help` lists the `run-<sample>` steps.

Overrides:

| Option | Default | Effect |
|---|---|---|
| `-Dgpu=sm_121` | `sm_80+ptx78` | The `.target` and `.version` of the PTX. The driver compiles `sm_80` PTX for any newer GPU. |
| `-Doptimize=fast` | `debug` | The optimization mode of the host programs. |
| `-Dkernel-optimize=safe` | `fast` | The optimization mode of the kernels. `safe` adds bounds and overflow checks that trap on the GPU. |
| `-Dtarget=x86_64-windows` | native | The host target. |
| `-Dcuda-bin=<dir>` | the directory of `nvcc` in `PATH` | The CUDA toolkit whose `nvcc`, `ptxas` and `nvdisasm` the `compare-nvcc` step runs. |

## Comparison with nvcc

    $ZIG_NVPTX_PREFIX/bin/zig build compare-nvcc

The step needs a CUDA toolkit. The build fetches the upstream samples (the `upstream` dependency
of `build.zig.zon`), and nvcc compiles each `.cu` file of each ported sample to PTX with the flags
of `nvcc_flags` in `build.zig` and the `-arch` that matches `-Dgpu`. One ptxas assembles the PTX of
both compilers for the `-Dgpu` target, and `nvcc_compare.zig` pairs each Zig kernel with the nvcc
kernel of the same name. `zig-out/nvcc-compare.md` gives the registers, stack frame, spill stores,
spill loads, shared memory and SASS instructions of each pair, a summary, and the kernels without a
partner. The step fails when a Zig kernel has no nvcc kernel, unless the upstream sample does not
instantiate it, or when a Zig kernel uses local memory (a stack frame or spill stores) and its nvcc
kernel does not. A static instruction count favors the compiler that unrolls less, so it measures
code size and not speed.

## Layout

| Path | Contents |
|---|---|
| `<category>/<sample>/` | One sample, in the folder of the upstream sample. `<sample>.zig` is the host program; each `*_kernel.zig` file becomes one PTX module. A folder with only a README.md marks a sample that is blocked or that waits for its port; `SAMPLES.md` gives the status. |
| `../../cuda.zig` | The shared host module `cuda`: the CUDA driver API, loaded from `libcuda.so.1` or `nvcuda.dll`. |
| `SAMPLES.md` | Every upstream sample with its category and port status. |
| `BACKEND-ISSUES.md`, `issues/` | Compiler problems that the ports meet. |
| `PORTING.md` | How to port a sample. |

The ported files keep the copyright notice of NVIDIA's original files; `LICENSE-cuda-samples` is the
license of the upstream repository.
