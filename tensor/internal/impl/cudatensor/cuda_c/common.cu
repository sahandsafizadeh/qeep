#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "types.h"
#include "common.cuh"

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

/* ----- indexing ----- */
size_t elemcnt(DimArr dims)
{
    size_t count = 1;
    for (size_t i = 0; i < dims.size; i++)
    {
        count *= dims.arr[i];
    }

    return count;
}