// nvcc -arch=sm_90 -c minmax_cvt_crash.cu   (ptxas 13.0 crashes)
__global__ void k(bool *out, unsigned x) {
    unsigned max;
    asm("max.bf16x2 %0, %1, %2;" : "=r"(max) : "r"(x), "r"(0x3F803F80u));
    unsigned short low;
    asm("cvt.u16.u32 %0, %1;" : "=h"(low) : "r"(max));
    *out = low == 0x3F80;
}
