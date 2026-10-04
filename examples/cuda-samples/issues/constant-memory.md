# Host-written `__constant__` variables

Samples: every sample with a `__constant__` array that the host fills with `cudaMemcpyToSymbol`
(for example convolutionSeparable, quasirandomGenerator, SobolQRNG).

Zig rejects a mutable variable in the `.constant` address space on nvptx, so a kernel cannot
declare a `.const` variable that the host writes before the launch:

```zig
export var table: [4]f32 addrspace(.constant) = undefined;
export fn read(out: [*]addrspace(.global) f32) callconv(.kernel) void {
    out[0] = table[0];
}
```

```
$ zig build-obj -target nvptx64-cuda -mcpu=sm_80 constant.zig
constant.zig:1:37: error: mutable values with address space 'constant' are not supported on nvptx
```

`std.Target.supportsAddressSpace` allows `.constant` only for `const` declarations, and Sema
reports the error. An `export const ... addrspace(.constant) = undefined` compiles, but the
compiler then folds every read of the undefined value, so a kernel never sees the bytes that the
host writes.

Workaround: declare the variable as `export var table: [4]f32 addrspace(.global) = undefined;`
(a `__device__` variable in CUDA C++) and fill it with `module.global(f32, "table")` and
`copyFrom`. The results stay the same; only the memory space of the loads changes.
