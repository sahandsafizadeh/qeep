#ifndef COMMON_H
#define COMMON_H

#define BLOCKS 256
#define THREADS 256

typedef double (*scalarUnaryFunc)(double);
typedef double (*scalarBinaryFunc)(double, double);

__device__ inline int getThreadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline int getGridStepSize()
{
    return gridDim.x * blockDim.x;
}

#endif // COMMON_H
