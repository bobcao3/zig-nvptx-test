# zig-nvptx-test

GPU tests and examples for Zig's self-hosted PTX backend (`nvptx64-cuda`).

Requires a Zig fork with the PTX backend, an NVIDIA GPU and driver, and `ptxas` in `PATH`.
Without a GPU, the tests still compile and validate PTX; without `ptxas`, they skip PTX validation.

## Usage

Run from the repository root with the fork's `zig`:

```sh
zig build                 # Install tools in zig-out/bin.
zig build test            # Compile, validate and run the GPU suites.
zig build behavior        # Run the Zig checkout's test/behavior on the GPU.
```

`behavior` writes `zig-out/behavior.md` and accepts file names after `--`.
Use `-Dzig-src=<checkout>` if the build cannot find the matching Zig sources.
See `zig build --help` for individual suites and options.

## Examples

Run `zig build` from `examples/cuda-samples/` or `examples/ml/` etc. to build all examples;
run the installed executables under `zig-out/bin/` directly.
