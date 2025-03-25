#include <stdio.h>

__global__ void kernel()
{
    printf("Hello from Device; Thread %d!\n", threadIdx.x);
}

extern "C"
{
    void cuda_hello()
    {
        kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
    }
}