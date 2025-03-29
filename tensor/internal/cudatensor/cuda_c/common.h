#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <cuda_runtime.h>

typedef struct LaunchParams
{
    int blockSize;
    int threadSize;
};

typedef double(scalarUnaryFunc)(double);
typedef double(scalarBinaryFunc)(double, double);

LaunchParams launchParams(size_t n);
void handleCudaError(cudaError_t err);

#endif // COMMON_H