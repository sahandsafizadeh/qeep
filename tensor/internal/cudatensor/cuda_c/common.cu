#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"

/* ----- kernel launch ----- */

const int MAX_THREADS_PER_BLOCK_X = 512;
const int MAX_BLOCKS_PER_GRID_X = 65535;

LaunchParams launchParams(size_t n)
{
    int threads = MAX_THREADS_PER_BLOCK_X;
    int blocks = (n + threads - 1) / threads;
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