#include <time.h>
#include <curand_kernel.h>

#include "types.h"
#include "common.cuh"

/* ----- helper functions ----- */

inline unsigned long long timeSeed()
{
    return (unsigned long long)time(NULL);
}

/* ----- device functions ----- */

__global__ void fillConst(CudaData dst, double value)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        dst.arr[i] = value;
    }
}

__global__ void fillEye(CudaData dst, size_t d)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        dst.arr[i] = i % (d + 1) == 0 ? 1. : 0.;
    }
}

__global__ void fillRandU(CudaData dst, double l, double u, unsigned long long seed)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tpos, 0, &state);

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        double randu_0_1 = curand_uniform_double(&state);
        dst.arr[i] = l + (u - l) * randu_0_1;
    }
}

__global__ void fillRandN(CudaData dst, double u, double s, unsigned long long seed)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tpos, 0, &state);

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        double randn_0_1 = curand_normal_double(&state);
        dst.arr[i] = u + s * randn_0_1;
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Full(size_t n, double value);
    double *Eye(size_t n, size_t d);
    double *RandU(size_t n, double l, double u);
    double *RandN(size_t n, double u, double s);
    double *Of(size_t n, double *input_data);
}

double *Full(size_t n, double value)
{
    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(dst.size);

    fillConst<<<lps.blockSize, lps.threadSize>>>(dst, value);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *Eye(size_t n, size_t d)
{
    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(dst.size);

    fillEye<<<lps.blockSize, lps.threadSize>>>(dst, d);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *RandU(size_t n, double l, double u)
{
    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    unsigned long long seed = timeSeed();

    LaunchParams lps = launchParams(dst.size);

    fillRandU<<<lps.blockSize, lps.threadSize>>>(dst, l, u, seed);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *RandN(size_t n, double u, double s)
{
    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    unsigned long long seed = timeSeed();

    LaunchParams lps = launchParams(dst.size);

    fillRandN<<<lps.blockSize, lps.threadSize>>>(dst, u, s, seed);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *Of(size_t n, double *input_data)
{
    double *dst;
    handleCudaError(
        cudaMalloc(&dst, n * sizeof(double)));

    handleCudaError(
        cudaMemcpy(
            dst,
            input_data,
            n * sizeof(double),
            cudaMemcpyHostToDevice));

    return dst;
}