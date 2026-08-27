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

__host__ __device__ inline size_t index2lnpos(DimArr index, CUDAView view)
{
    size_t lnpos = view.ofst;
    for (int i = 0; i < index.size; i++)
    {
        lnpos += view.strd.arr[i] * index.arr[i];
    }

    return lnpos;
}

__host__ __device__ inline DimArr lnpos2index(size_t lnpos, CUDAView view)
{
    DimArr index;
    index.size = view.dims.size;

    size_t rem = lnpos - view.ofst;
    for (int i = 0; i < view.strd.size; i++)
    {
        size_t count = view.strd.arr[i];
        index.arr[i] = rem / count;
        rem = rem % count;
    }

    return index;
}

__host__ __device__ inline size_t flatpos(size_t i, CUDAView view)
{
    DimArr index;
    index.size = view.dims.size;

    size_t rem = i;
    for (int j = index.size - 1; j >= 0; j--)
    {
        size_t dim = view.dims.arr[j];
        index.arr[j] = rem % dim;
        rem /= dim;
    }

    return index2lnpos(index, view);
}

#endif // DEVCOMMON_H