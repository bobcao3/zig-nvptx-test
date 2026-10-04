# cuda-samples classification

Every sample under `cpp/` of [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/cpp).

| Category | Meaning |
|---|---|
| PORT | The sample has kernels whose CUDA C++ device code a Zig kernel can express. |
| SKIP-runtime | The sample demonstrates a runtime or driver API feature; its kernels are trivial. |
| SKIP-library | The sample wraps a CUDA library (cuBLAS, cuFFT, cuRAND, Thrust, CUB, libcu++). |
| SKIP-graphics | The sample needs OpenGL, EGL, Vulkan or Direct3D. |
| SKIP-other | The sample needs NVRTC, libNVVM, nvJitLink, NvSci, CUDA Tile or a Tegra device. |

| Status | Meaning |
|---|---|
| PORTED | The Zig port passes on the GPU. |
| TODO | The sample waits for its port. |
| BLOCKED | The port needs the feature in the reason column first. The folder holds only a README.md. |
| SKIPPED | The sample stays out of the mirror. |

Counts: PORT 65, SKIP-runtime 36, SKIP-library 16, SKIP-graphics 28, SKIP-other 31 (176 samples). PORT status: PORTED 56, BLOCKED 8, TODO 1.

| Sample | Category | Status | Reason |
|---|---|---|---|
| 0_Introduction/asyncAPI | SKIP-runtime | SKIPPED | events and CPU/GPU overlap |
| 0_Introduction/clock | PORT | PORTED | `clock()` timing per block, shared-memory reduction |
| 0_Introduction/cudaOpenMP | SKIP-runtime | SKIPPED | OpenMP host threads per GPU |
| 0_Introduction/fp16ScalarProduct | PORT | PORTED | `half2` arithmetic |
| 0_Introduction/matrixMul | PORT | PORTED | shared-memory tiled GEMM |
| 0_Introduction/matrixMulDrv | PORT | PORTED | driver API tiled GEMM |
| 0_Introduction/matrixMulDynlinkJIT | SKIP-runtime | SKIPPED | dynamic driver loading and JIT, which cuda.zig does for every sample |
| 0_Introduction/simpleAssert | PORT | PORTED | device-side `assert` |
| 0_Introduction/simpleAtomicIntrinsics | PORT | PORTED | global and shared atomics |
| 0_Introduction/simpleAttributes | SKIP-runtime | SKIPPED | stream L2 access-policy attributes |
| 0_Introduction/simpleAWBarrier | PORT | PORTED | arrive-wait barrier and grid-wide sync |
| 0_Introduction/simpleCallback | SKIP-runtime | SKIPPED | host callbacks in streams |
| 0_Introduction/simpleCooperativeGroups | PORT | PORTED | thread-block groups and tiled partitions |
| 0_Introduction/simpleCubemapTexture | PORT | PORTED | cubemap texture lookups |
| 0_Introduction/simpleCUDA2GL | SKIP-graphics | SKIPPED | OpenGL |
| 0_Introduction/simpleDrvRuntime | SKIP-runtime | SKIPPED | driver and runtime API interplay |
| 0_Introduction/simpleIPC | SKIP-runtime | SKIPPED | inter-process memory handles |
| 0_Introduction/simpleLayeredTexture | PORT | PORTED | layered 2D texture lookups |
| 0_Introduction/simpleMPI | SKIP-runtime | SKIPPED | MPI |
| 0_Introduction/simpleMultiCopy | SKIP-runtime | SKIPPED | copy/compute overlap |
| 0_Introduction/simpleMultiGPU | SKIP-runtime | SKIPPED | multiple GPUs |
| 0_Introduction/simpleP2P | SKIP-runtime | SKIPPED | peer-to-peer copies |
| 0_Introduction/simplePitchLinearTexture | PORT | PORTED | pitch-linear and array textures |
| 0_Introduction/simplePrintf | PORT | PORTED | device-side `printf` (`vprintf`) |
| 0_Introduction/simpleStreams | SKIP-runtime | SKIPPED | streams |
| 0_Introduction/simpleSurfaceWrite | PORT | PORTED | surface writes and texture reads |
| 0_Introduction/simpleTexture | PORT | PORTED | 2D texture rotation of an image |
| 0_Introduction/simpleTexture3D | PORT | PORTED | 3D texture slice with normalized reads |
| 0_Introduction/simpleTextureDrv | PORT | PORTED | 2D texture rotation through the driver API |
| 0_Introduction/simpleVoteIntrinsics | PORT | PORTED | `__any_sync`, `__all_sync`, `__ballot_sync` |
| 0_Introduction/simpleZeroCopy | SKIP-runtime | SKIPPED | mapped host memory |
| 0_Introduction/systemWideAtomics | PORT | PORTED | system-scope atomics on managed memory |
| 0_Introduction/UnifiedMemoryStreams | SKIP-runtime | SKIPPED | OpenMP, streams and unified memory |
| 0_Introduction/vectorAdd | PORT | PORTED | element-wise kernel, unified memory |
| 0_Introduction/vectorAdd_nvrtc | SKIP-other | SKIPPED | NVRTC compiles CUDA C++ at run time |
| 0_Introduction/vectorAddDrv | PORT | PORTED | driver API vector addition |
| 0_Introduction/vectorAddMMAP | SKIP-runtime | SKIPPED | virtual memory management (`cuMemMap`) |
| 1_Utilities/deviceQuery | SKIP-runtime | SKIPPED | device properties |
| 1_Utilities/deviceQueryDrv | SKIP-runtime | SKIPPED | device properties |
| 1_Utilities/topologyQuery | SKIP-runtime | SKIPPED | GPU topology |
| 2_Concepts_and_Techniques/boxFilter | SKIP-graphics | SKIPPED | OpenGL |
| 2_Concepts_and_Techniques/convolutionSeparable | PORT | PORTED | `__constant__` filter, shared-memory tiles |
| 2_Concepts_and_Techniques/convolutionTexture | PORT | PORTED | separable convolution through a texture |
| 2_Concepts_and_Techniques/dct8x8 | PORT | PORTED | 8x8 discrete cosine transform |
| 2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU | SKIP-graphics | SKIPPED | EGL |
| 2_Concepts_and_Techniques/EGLStream_CUDA_Interop | SKIP-graphics | SKIPPED | EGL |
| 2_Concepts_and_Techniques/eigenvalues | PORT | BLOCKED | a safety check traps in safe mode; the cause is not known |
| 2_Concepts_and_Techniques/FunctionPointers | SKIP-graphics | SKIPPED | OpenGL |
| 2_Concepts_and_Techniques/histogram | PORT | PORTED | shared-memory atomics, 64 and 256 bins |
| 2_Concepts_and_Techniques/imageDenoising | SKIP-graphics | SKIPPED | OpenGL |
| 2_Concepts_and_Techniques/inlinePTX | PORT | PORTED | inline PTX (`%laneid`) |
| 2_Concepts_and_Techniques/interval | PORT | PORTED | interval arithmetic, templates and device recursion |
| 2_Concepts_and_Techniques/MC_EstimatePiInlineP | SKIP-library | SKIPPED | cuRAND |
| 2_Concepts_and_Techniques/MC_EstimatePiInlineQ | SKIP-library | SKIPPED | cuRAND |
| 2_Concepts_and_Techniques/MC_EstimatePiP | SKIP-library | SKIPPED | cuRAND |
| 2_Concepts_and_Techniques/MC_EstimatePiQ | SKIP-library | SKIPPED | cuRAND |
| 2_Concepts_and_Techniques/MC_SingleAsianOptionP | SKIP-library | SKIPPED | cuRAND |
| 2_Concepts_and_Techniques/particles | SKIP-graphics | SKIPPED | OpenGL |
| 2_Concepts_and_Techniques/radixSortThrust | SKIP-library | SKIPPED | Thrust |
| 2_Concepts_and_Techniques/reduction | PORT | PORTED | seven shared-memory and shuffle reduction kernels |
| 2_Concepts_and_Techniques/reductionMultiBlockCG | PORT | PORTED | single-kernel reduction with a grid-wide barrier |
| 2_Concepts_and_Techniques/scalarProd | PORT | PORTED | shared-memory dot products |
| 2_Concepts_and_Techniques/scan | PORT | PORTED | shared-memory prefix sum |
| 2_Concepts_and_Techniques/segmentationTreeThrust | SKIP-library | SKIPPED | Thrust |
| 2_Concepts_and_Techniques/shfl_scan | PORT | PORTED | `__shfl_up_sync` scan and integral image |
| 2_Concepts_and_Techniques/sortingNetworks | PORT | PORTED | bitonic and odd-even merge sort |
| 2_Concepts_and_Techniques/streamOrderedAllocation | SKIP-runtime | SKIPPED | `cudaMallocAsync` pools |
| 2_Concepts_and_Techniques/streamOrderedAllocationIPC | SKIP-runtime | SKIPPED | memory pool IPC |
| 2_Concepts_and_Techniques/streamOrderedAllocationP2P | SKIP-runtime | SKIPPED | memory pool peer access |
| 2_Concepts_and_Techniques/threadFenceReduction | PORT | PORTED | single-pass reduction with `__threadfence` |
| 2_Concepts_and_Techniques/threadMigration | SKIP-runtime | SKIPPED | context management across host threads |
| 3_CUDA_Features/bf16TensorCoreGemm | PORT | PORTED | WMMA bf16 GEMM with `cp.async` |
| 3_CUDA_Features/binaryPartitionCG | PORT | PORTED | cooperative groups `labeled_partition` and warp reduce |
| 3_CUDA_Features/bindlessTexture | SKIP-graphics | SKIPPED | OpenGL |
| 3_CUDA_Features/cdpAdvancedQuicksort | PORT | BLOCKED | dynamic parallelism (device-side kernel launch through `cudadevrt`) |
| 3_CUDA_Features/cdpBezierTessellation | PORT | BLOCKED | dynamic parallelism (device-side kernel launch through `cudadevrt`) |
| 3_CUDA_Features/cdpQuadtree | PORT | BLOCKED | dynamic parallelism (device-side kernel launch through `cudadevrt`) |
| 3_CUDA_Features/cdpSimplePrint | PORT | BLOCKED | dynamic parallelism (device-side kernel launch through `cudadevrt`) |
| 3_CUDA_Features/cdpSimpleQuicksort | PORT | BLOCKED | dynamic parallelism (device-side kernel launch through `cudadevrt`) |
| 3_CUDA_Features/cudaCompressibleMemory | SKIP-runtime | SKIPPED | compressible memory allocation |
| 3_CUDA_Features/cudaTensorCoreGemm | PORT | PORTED | WMMA f16 GEMM (`mma.sync` in inline PTX) |
| 3_CUDA_Features/dmabufInterop | SKIP-runtime | SKIPPED | dma-buf interop |
| 3_CUDA_Features/dmmaTensorCoreGemm | PORT | PORTED | WMMA f64 GEMM |
| 3_CUDA_Features/globalToShmemAsyncCopy | PORT | PORTED | `memcpy_async`, `cuda::pipeline`, arrive-wait barriers |
| 3_CUDA_Features/graphConditionalNodes | SKIP-runtime | SKIPPED | CUDA graphs |
| 3_CUDA_Features/graphMemoryFootprint | SKIP-runtime | SKIPPED | CUDA graphs |
| 3_CUDA_Features/graphMemoryNodes | SKIP-runtime | SKIPPED | CUDA graphs |
| 3_CUDA_Features/immaTensorCoreGemm | PORT | PORTED | WMMA int8 GEMM |
| 3_CUDA_Features/jacobiCudaGraphs | SKIP-runtime | SKIPPED | CUDA graphs |
| 3_CUDA_Features/localityDomains | SKIP-runtime | SKIPPED | locality domain queries |
| 3_CUDA_Features/localityDomainsDrv | SKIP-runtime | SKIPPED | locality domain queries |
| 3_CUDA_Features/memMapIPCDrv | SKIP-runtime | SKIPPED | virtual memory IPC |
| 3_CUDA_Features/newdelete | PORT | PORTED | device heap (`malloc`/`new` in kernels) |
| 3_CUDA_Features/ptxjit | SKIP-runtime | SKIPPED | PTX JIT linking, which cuda.zig does for every sample |
| 3_CUDA_Features/simpleCudaGraphs | SKIP-runtime | SKIPPED | CUDA graphs |
| 3_CUDA_Features/StreamPriorities | SKIP-runtime | SKIPPED | stream priorities |
| 3_CUDA_Features/tf32TensorCoreGemm | PORT | PORTED | WMMA tf32 GEMM with `cp.async` |
| 3_CUDA_Features/warpAggregatedAtomicsCG | PORT | PORTED | warp-aggregated atomics with coalesced groups |
| 4_CUDA_Libraries/conjugateGradientCudaGraphs | SKIP-library | SKIPPED | cuBLAS, cuSPARSE |
| 4_CUDA_Libraries/conjugateGradientMultiBlockCG | PORT | PORTED | conjugate gradient solver with a grid-wide barrier |
| 4_CUDA_Libraries/conjugateGradientMultiDeviceCG | PORT | BLOCKED | multi-device grid-wide barrier (`multi_grid_group`) is not in std.nvptx |
| 4_CUDA_Libraries/cubDeviceFind | SKIP-library | SKIPPED | CUB |
| 4_CUDA_Libraries/cubDeviceSegmentedScan | SKIP-library | SKIPPED | CUB |
| 4_CUDA_Libraries/cubDeviceTransform | SKIP-library | SKIPPED | CUB |
| 4_CUDA_Libraries/cudaNvSci | SKIP-other | SKIPPED | NvSci (Tegra and driver libraries) |
| 4_CUDA_Libraries/jitLto | SKIP-other | SKIPPED | NVRTC and nvJitLink |
| 4_CUDA_Libraries/libcuxxMdspan | SKIP-library | SKIPPED | libcu++ |
| 4_CUDA_Libraries/libcuxxRandom | SKIP-library | SKIPPED | libcu++ |
| 4_CUDA_Libraries/lineOfSight | SKIP-library | SKIPPED | Thrust |
| 4_CUDA_Libraries/oceanFFT | SKIP-graphics | SKIPPED | OpenGL, cuFFT |
| 4_CUDA_Libraries/randomFog | SKIP-graphics | SKIPPED | OpenGL, cuRAND |
| 5_Domain_Specific/bicubicTexture | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/bilateralFilter | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/binomialOptions | PORT | PORTED | binomial option pricing, shared memory |
| 5_Domain_Specific/BlackScholes | PORT | PORTED | option pricing, math functions |
| 5_Domain_Specific/convolutionFFT2D | SKIP-library | SKIPPED | cuFFT |
| 5_Domain_Specific/dwtHaar1D | PORT | PORTED | Haar wavelet, reads a data file |
| 5_Domain_Specific/dxtc | PORT | PORTED | DXT1 compression, reads an image file |
| 5_Domain_Specific/fastWalshTransform | PORT | PORTED | shared-memory butterfly transform |
| 5_Domain_Specific/FDTD3d | PORT | PORTED | 3D finite-difference stencil |
| 5_Domain_Specific/fluidsGL | SKIP-graphics | SKIPPED | OpenGL, cuFFT |
| 5_Domain_Specific/HSOpticalFlow | PORT | PORTED | Horn-Schunck optical flow |
| 5_Domain_Specific/Mandelbrot | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/marchingCubes | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/MonteCarloMultiGPU | SKIP-library | SKIPPED | cuRAND |
| 5_Domain_Specific/nbody | PORT | PORTED | all-pairs n-body (benchmark and compare modes, no OpenGL) |
| 5_Domain_Specific/NV12toBGRandResize | PORT | PORTED | NV12 to BGR conversion and resize |
| 5_Domain_Specific/postProcessGL | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/quasirandomGenerator | PORT | PORTED | Niederreiter sequence, inverse CND |
| 5_Domain_Specific/recursiveGaussian | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/simpleD3D11 | SKIP-graphics | SKIPPED | Direct3D 11 |
| 5_Domain_Specific/simpleD3D11Texture | SKIP-graphics | SKIPPED | Direct3D 11 |
| 5_Domain_Specific/simpleD3D12 | SKIP-graphics | SKIPPED | Direct3D 12 |
| 5_Domain_Specific/simpleGL | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/simpleVulkan | SKIP-graphics | SKIPPED | Vulkan |
| 5_Domain_Specific/simpleVulkanMMAP | SKIP-graphics | SKIPPED | Vulkan |
| 5_Domain_Specific/smokeParticles | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/SobelFilter | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/SobolQRNG | PORT | PORTED | Sobol sequence (large direction-number table) |
| 5_Domain_Specific/stereoDisparity | PORT | BLOCKED | `std.nvptx` lacks `vabsdiff4` (CUDA `__usad4`) |
| 5_Domain_Specific/volumeFiltering | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/volumeRender | SKIP-graphics | SKIPPED | OpenGL |
| 5_Domain_Specific/vulkanImageCUDA | SKIP-graphics | SKIPPED | Vulkan |
| 6_Performance/alignedTypes | PORT | PORTED | aligned struct loads and stores |
| 6_Performance/cudaGraphsPerfScaling | SKIP-runtime | SKIPPED | CUDA graphs |
| 6_Performance/LargeKernelParameter | PORT | TODO | a 32000-byte kernel parameter, which needs PTX ISA 8.1 |
| 6_Performance/transpose | PORT | PORTED | eight shared-memory transpose kernels |
| 6_Performance/UnifiedMemoryPerf | SKIP-runtime | SKIPPED | unified memory performance |
| 7_libNVVM/cuda-c-linking | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 7_libNVVM/cuda-shared-memory | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 7_libNVVM/device-side-launch | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 7_libNVVM/ptxgen | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 7_libNVVM/simple | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 7_libNVVM/syscalls | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 7_libNVVM/uvmlite | SKIP-other | SKIPPED | libNVVM compiles NVVM IR |
| 8_Platform_Specific/Tegra/EGLSync_CUDAEvent_Interop | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cuDLAErrorReporting | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cuDLAHybridMode | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cuDLALayerwiseStatsStandalone | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cuDLAStandaloneMode | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/cudaNvSciNvMedia | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/fluidsGLES | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/nbody_opengles | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/simpleGLES | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 8_Platform_Specific/Tegra/simpleGLES_EGLOutput | SKIP-other | SKIPPED | Tegra-only (cuDLA, NvMedia, EGL, OpenGL ES) |
| 9_CUDA_Tile/helloTile | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileBmm | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileLayerNorm | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileMatmul | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileMatmulAutotuner | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileRope | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileSpMV | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileTranspose | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
| 9_CUDA_Tile/tileVectorAdd | SKIP-other | SKIPPED | CUDA Tile programming model, not PTX |
