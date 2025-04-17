#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <cuda_runtime.h>

const unsigned int MAX_THREADS_PER_BLOCK_X = 512;
const unsigned int MAX_BLOCKS_PER_GRID_X = 65535;

typedef struct LaunchParams
{
    unsigned int blockSize;
    unsigned int threadSize;
} LaunchParams;

LaunchParams launchParams(size_t n);
void handleCudaError(cudaError_t err);

#endif // COMMON_H