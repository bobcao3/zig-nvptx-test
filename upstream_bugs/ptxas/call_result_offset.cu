// nvcc -arch=sm_121 call_result_offset.cu -o call_result_offset && ./call_result_offset
#include <cstdio>
__device__ __noinline__ long long f(long long x) { return x + 1; }
__global__ void k(long long *w) {
    w[0] = f(1) + (1ll << 32);
    w[1] = f(2) + (1ll << 32);
}
int main() {
    long long *w, h[2];
    cudaMalloc(&w, sizeof h);
    k<<<1, 1>>>(w);
    cudaMemcpy(h, w, sizeof h, cudaMemcpyDeviceToHost);
    printf("%llx %llx (want 100000002 100000003)\n", h[0], h[1]);
}
