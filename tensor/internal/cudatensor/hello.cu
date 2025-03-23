#include <stdio.h>

__global__ void kernel()
{
    printf("Hello from Device Thread %d!\n", threadIdx.x);
}

int main()
{
    printf("Hello from Host!\n");
    kernel<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}