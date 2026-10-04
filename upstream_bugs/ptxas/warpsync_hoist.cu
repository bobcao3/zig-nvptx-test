// nvcc -arch=sm_121 warpsync_hoist.cu -o warpsync_hoist && timeout 15 ./warpsync_hoist   (hangs)
#include <cstdio>
__global__ void k() {
    __shared__ unsigned lock, counter;
    if (threadIdx.x == 0) lock = counter = 0;
    __syncthreads();
    while (atomicCAS(&lock, 0, 1) != 0) {}
    counter += 1;
    *(volatile unsigned *)&lock = 0;
    __syncthreads();
}
int main() {
    k<<<1, 32>>>();
    printf("%s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}
