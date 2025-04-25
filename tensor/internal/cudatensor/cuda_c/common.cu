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

int elemcnt(DimArr dims)
{
    int count = 1;
    for (size_t i = 0; i < dims.size; i++)
    {
        count *= dims.arr[i];
    }

    return count;
}

int elemcnt(RangeArr ranges)
{
    int count = 1;
    for (size_t i = 0; i < ranges.size; i++)
    {
        Range range = ranges.arr[i];
        count *= range.to - range.from;
    }

    return count;
}

DimArr rcumprod(DimArr dims)
{
    DimArr rcp;

    int prod = 1;
    for (size_t i = 0; i < dims.size; i++)
    {
        size_t j = dims.size - i - 1;
        rcp.arr[j] = prod;
        prod *= dims.arr[j];
    }

    rcp.size = dims.size;

    return rcp;
}

DimArr rcumprod(RangeArr ranges)
{
    DimArr rcp;

    int prod = 1;
    for (size_t i = 0; i < ranges.size; i++)
    {
        size_t j = ranges.size - i - 1;
        Range range = ranges.arr[j];
        rcp.arr[j] = prod;
        prod *= range.to - range.from;
    }

    rcp.size = ranges.size;

    return rcp;
}