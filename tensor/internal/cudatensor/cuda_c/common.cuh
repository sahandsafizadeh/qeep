#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <cuda_runtime.h>

#include "types.h"

/* ---------- kernel launch ---------- */
const unsigned int MAX_BLOCKS_PER_GRID_X = 65535;
const unsigned int MAX_THREADS_PER_BLOCK_X = 512;

typedef struct LaunchParams
{
    unsigned int blockSize;
    unsigned int threadSize;
} LaunchParams;

LaunchParams launchParams(size_t n);

/* --------- error handling ---------- */
void handleCudaError(cudaError_t err);

/* ----------------------------------- */
/* ------------ indexing ------------- */
/* ----------------------------------- */

int elemcnt(DimArr dims);
int elemcnt(RangeArr ranges);
DimArr rcumprod(DimArr dims);
DimArr rcumprod(RangeArr ranges);

__device__ inline unsigned int blockIndex()
{
    return blockIdx.x;
}

__device__ inline unsigned int threadIndex()
{
    return threadIdx.x;
}

__device__ inline unsigned int blockSize()
{
    return blockDim.x;
}

__device__ inline unsigned int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__device__ inline unsigned int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__host__ __device__ inline int encode(DimArr index, DimArr rcp)
{
    int lnpos = 0;
    for (size_t i = 0; i < rcp.size; i++)
    {
        lnpos += index.arr[i] * rcp.arr[i];
    }
    return lnpos;
}

__host__ __device__ inline DimArr decode(int lnpos, DimArr rcp)
{
    DimArr index;
    int rem = lnpos;
    for (size_t i = 0; i < rcp.size; i++)
    {
        int count = rcp.arr[i];
        index.arr[i] = rem / count;
        rem = rem % count;
    }
    index.size = rcp.size;
    return index;
}

#endif // COMMON_H