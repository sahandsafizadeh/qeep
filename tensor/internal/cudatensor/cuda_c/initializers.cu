#include <time.h>
#include <curand_kernel.h>

#include "common.h"

/* ----- helper functions ----- */

inline unsigned long long timeSeed()
{
    return (unsigned long long)time(NULL);
}

/* ----- device functions ----- */

__device__ inline unsigned int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline unsigned int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__global__ void fillConst(size_t n, double value, double *data)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        data[i] = value;
    }
}

__global__ void fillEye(size_t n, size_t d, double *data)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        data[i] = i % (d + 1) == 0 ? 1. : 0.;
    }
}

__global__ void fillRandU(size_t n, double l, double u, unsigned long long seed, double *data)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tpos, 0, &state);

    for (size_t i = tpos; i < n; i += stride)
    {
        double randu_0_1 = curand_uniform_double(&state);
        data[i] = l + (u - l) * randu_0_1;
    }
}

__global__ void fillRandN(size_t n, double u, double s, unsigned long long seed, double *data)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tpos, 0, &state);

    for (size_t i = tpos; i < n; i += stride)
    {
        double randn_0_1 = curand_normal_double(&state);
        data[i] = u + s * randn_0_1;
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Full(size_t n, double value);
    double *Eye(size_t n, size_t d);
    double *RandU(size_t n, double l, double u);
    double *RandN(size_t n, double u, double s);
    double *Of(const double *input_data, size_t n);
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

double *Eye(size_t n, size_t d)
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

    unsigned long long seed = timeSeed();

    LaunchParams lps = launchParams(n);

    fillRandU<<<lps.blockSize, lps.threadSize>>>(n, l, u, seed, data);

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

    unsigned long long seed = timeSeed();

    LaunchParams lps = launchParams(n);

    fillRandN<<<lps.blockSize, lps.threadSize>>>(n, u, s, seed, data);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return data;
}

double *Of(const double *input_data, size_t n)
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