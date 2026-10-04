# eigenvalues: trap in a bisection kernel in safe mode

Samples: eigenvalues. The sample is blocked, and its folder holds only `README.md`.

A port of the three large-matrix bisection kernels (`bisectKernelLarge`,
`bisectKernelLargeOneIntervals`, `bisectKernelLargeMultIntervals`) and the small-matrix kernel
compiled and passed ptxas. With the default `-Doptimize=safe` the run failed; with
`-Doptimize=fast` it printed `Test Succeeded!`. The output below is the output of that port.

```
$ zig build run-eigenvalues
Starting eigenvalues
Matrix size: 2048 x 2048
Gerschgorin interval: -2.894311 / 2.923303
error: CUDA driver: CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
error: LaunchFailed

$ compute-sanitizer zig-out/bin/2_Concepts_and_Techniques/eigenvalues/eigenvalues
========= Trace/breakpoint trap
=========     at bisectKernelLargeMultIntervals+0x37f0
=========     by thread (7,0,0) in block (0,0,0)

$ zig build run-eigenvalues -Doptimize=fast
Average time TOTAL: 6.272878 ms
Test Succeeded!
```

The trap came from a safety check of a kernel in the PTX (`trap` instruction). The check that fired
was not identified: unsigned subtractions in the kernels used wrapping operators as in the C++ code,
and the trap remained. It is not known whether the port or the backend was at fault. A new port is
needed to check the current compiler.
