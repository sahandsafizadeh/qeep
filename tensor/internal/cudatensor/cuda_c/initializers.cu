#include "common.h"
#include "goutil.h"

/* ----- device functions ----- */

__device__ inline int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__global__ void fillConst(size_t n, double value, double *data)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        data[i] = value;
    }
}

__global__ void fillEye(size_t n, int d, double *data)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        data[i] = i % (d + 1) == 0 ? 1. : 0.;
    }
}

__global__ void fillRandU(size_t n, double l, double u, double *data)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        data[i] = goUniformRand(l, u);
    }
}

__global__ void fillRandN(size_t n, double u, double s, double *data)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        data[i] = goNormalRand(u, s);
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Full(size_t n, double value);
    double *Eye(size_t n, int d);
    double *RandU(size_t n, double l, double u);
    double *RandN(size_t n, double u, double s);
    double *Of(size_t n, const double input_data[]);
}

double *Full(size_t n, double value)
{
    double *data;
    handleCudaError(
        cudaMalloc(&data, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    fillConst<<<lps.blockSize, lps.threadSize>>>(n, value, data);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return data;
}

double *Eye(size_t n, int d)
{
    double *data;
    handleCudaError(
        cudaMalloc(&data, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    fillEye<<<lps.blockSize, lps.threadSize>>>(n, d, data);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return data;
}

double *RandU(size_t n, double l, double u)
{
    double *data;
    handleCudaError(
        cudaMalloc(&data, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    fillRandU<<<lps.blockSize, lps.threadSize>>>(n, l, u, data);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return data;
}

double *RandN(size_t n, double u, double s)
{
    double *data;
    handleCudaError(
        cudaMalloc(&data, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    fillRandN<<<lps.blockSize, lps.threadSize>>>(n, u, s, data);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return data;
}

double *Of(size_t n, const double input_data[])
{
    double *data;
    handleCudaError(
        cudaMalloc(&data, n * sizeof(double)));

    handleCudaError(
        cudaMemcpy(
            data,
            input_data,
            n * sizeof(double),
            cudaMemcpyHostToDevice));

    return data;
}