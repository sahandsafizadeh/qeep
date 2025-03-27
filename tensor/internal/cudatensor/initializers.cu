#include "cudatensor.h"

/* ----- device functions ----- */

__device__ int getThreadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ int getGridStepSize()
{
    return gridDim.x * blockDim.x;
}

__global__ void fillTensorData(double *d, double value, size_t n)
{
    const int tp = getThreadPosition();
    const int gss = getGridStepSize();

    for (size_t i = tp; i < n; i += gss)
    {
        d[i] = value;
    }
}

/* ----- API functions ----- */

double *Full(size_t n, double value)
{
    double *d;
    cudaMalloc(&d, n * sizeof(double));
    fillTensorData<<<BLOCKS, THREADS>>>(d, value, n);
    return d;
}