# Backend issues

Each issue that a port meets has its own file in `issues/`, with a minimal repro, the command that
shows the problem and the workaround (or the samples that it blocks).

| Issue | Samples | Workaround |
|---|---|---|
| [Host-written `__constant__` variables](issues/constant-memory.md) | samples with `cudaMemcpyToSymbol` | `export var ... addrspace(.global)` |
| [eigenvalues: trap in a bisection kernel in safe mode](issues/eigenvalues-safe-mode-trap.md) | eigenvalues | none |
