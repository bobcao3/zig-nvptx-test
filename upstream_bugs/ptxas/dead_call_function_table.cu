// nvcc -arch=sm_121 -cubin dead_call_function_table.cu   (ptxas crashes)
__device__ __noinline__ void g() { asm volatile("trap;"); }
__device__ void f() {
    long long x;
    asm("mov.b64 %0, 0;" : "=l"(x));
    if ((x & 0xf) == 3) g();
}
__global__ void k(void (**p)()) { *p = f; }
