// nvcc -arch=sm_121 divergent_division.cu -o divergent_division && ./divergent_division
#include <cstdio>
__device__ __noinline__ void g() { asm volatile(""); }
__global__ void k(unsigned *out, unsigned n, unsigned d) {
    unsigned t = threadIdx.x, h = 0, prev = 0;
    for (unsigned i = 0; i < n; i++) {
        if (t == 0) {
            g();
            h ^= prev;
        } else {
            h = t / d;
        }
        g();
        prev = i;
    }
    out[t] = h;
}
int main() {
    unsigned *out, h;
    cudaMalloc(&out, 32 * sizeof h);
    k<<<1, 32>>>(out, 3, 7);
    cudaMemcpy(&h, out, sizeof h, cudaMemcpyDeviceToHost);
    printf("out[0] = %u (want 1)\n", h);
}
