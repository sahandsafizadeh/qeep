#include <stdlib.h>
#include <stdio.h>
#include <math.h>

const int MAX_THREADS_PER_BLOCK_X = 512;
const int MAX_BLOCKS_PER_GRID_X = 65535;

int threadSize(size_t n)
{
    return MAX_THREADS_PER_BLOCK_X;
}

int blockSize(size_t n)
{
    int threads = threadSize(n);
    int blocks = (n + threads - 1) / threads;
    return min(blocks, MAX_BLOCKS_PER_GRID_X);
}

void handleCudaError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s: %s\n",
               cudaGetErrorName(err),
               cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}