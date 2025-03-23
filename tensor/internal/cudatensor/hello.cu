#include <stdio.h>

__global__ void kernel()
{
    printf("Hello from Device; Block: %d, Thread %d!\n", blockIdx.x, threadIdx.x);
}

extern "C" void cuda_hello()
{
    kernel<<<5, 5>>>();
    cudaDeviceSynchronize();
}