#include "common.h"

/* ----- device functions ----- */

__global__ void fillDataWithValue(double *data, size_t n, double value)
{
    const int tpos = getThreadPosition();
    const int gstep = getGridStepSize();

    for (size_t i = tpos; i < n; i += gstep)
    {
        data[i] = value;
    }
}

/* ----- API functions ----- */

double *Full(size_t n, double value)
{
    double *dev_data;
    cudaMalloc(&dev_data, n * sizeof(double));

    fillDataWithValue<<<BLOCKS, THREADS>>>(dev_data, n, value);
    cudaDeviceSynchronize();

    return dev_data;
}

void FreeCUDAMemory(double *dev_data)
{
    cudaFree(dev_data);
}