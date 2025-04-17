#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"

/* ----- kernel launch ----- */

LaunchParams launchParams(size_t n)
{
    unsigned int threads = MAX_THREADS_PER_BLOCK_X;
    unsigned int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, MAX_BLOCKS_PER_GRID_X);
    return (LaunchParams){blocks, threads};
}

/* ----- error handling ----- */

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