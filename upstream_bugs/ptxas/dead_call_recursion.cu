// nvcc -arch=sm_121 -cubin dead_call_recursion.cu   (ptxas crashes)
__device__ __noinline__ void g() { asm volatile("trap;"); }
__device__ void h(int n) {
    long long x;
    asm("mov.b64 %0, 0;" : "=l"(x));
    if ((x & 0xf) == 3) g();
    if (n) {
        h(n - 1);
        h(n - 1);
    }
}
__global__ void k(int n) { h(n); }
