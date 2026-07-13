#ifndef DEVCOMMON_H
#define DEVCOMMON_H

#include <stddef.h>
#include "types.h"

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

__host__ __device__ inline int index2lnpos(DimArr index, CUDAView view)
{
    int lnpos = view.ofst;
    for (size_t i = 0; i < index.size; i++)
    {
        lnpos += view.strd.arr[i] * index.arr[i];
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

#endif // DEVCOMMON_H