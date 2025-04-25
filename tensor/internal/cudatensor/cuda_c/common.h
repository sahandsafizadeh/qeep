#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <cuda_runtime.h>

#include "types.h"

/* ---------- kernel launch ---------- */
const unsigned int MAX_THREADS_PER_BLOCK_X = 512;
const unsigned int MAX_BLOCKS_PER_GRID_X = 65535;
typedef struct LaunchParams
{
    unsigned int blockSize;
    unsigned int threadSize;
} LaunchParams;
LaunchParams launchParams(size_t n);

/* --------- error handling ---------- */
void handleCudaError(cudaError_t err);

/* ------------ indexing ------------- */
int elemcnt(DimArr dims);
int elemcnt(RangeArr ranges);
DimArr rcumprod(DimArr dims);
DimArr rcumprod(RangeArr ranges);

#endif // COMMON_H