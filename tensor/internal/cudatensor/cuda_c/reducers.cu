#include <math.h>
#include "common.h"

/* ----- reducer functions ----- */

double sum_(double a, double b)
{
    return a + b;
}

double max_(double a, double b)
{
    return a >= b ? a : b;
}

double min_(double a, double b)
{
    return a <= b ? a : b;
}

/* ----- device functions ----- */

typedef double(reducerFunc)(double, double);

__device__ inline unsigned int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline unsigned int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__device__ inline unsigned int threadIndex()
{
    return threadIdx.x;
}

__device__ inline unsigned int blockIndex()
{
    return blockIdx.x;
}

__device__ inline unsigned int blockSize()
{
    return blockDim.x;
}

__global__ void reduceByAssociativeFunc(
    double *dst,
    const double *src,
    size_t n,
    reducerFunc af,
    double identity)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    double temp = identity;
    for (size_t i = tpos; i < n; i += stride)
    {
        temp = af(temp, src[i]);
    }

    const unsigned int cacheidx = threadIndex();
    const unsigned int cachelen = blockSize();
    const unsigned int blockidx = blockIndex();

    __shared__ double cache[MAX_THREADS_PER_BLOCK_X];

    cache[cacheidx] = temp;
    __syncthreads();

    for (size_t i = cachelen / 2; i != 0; i /= 2)
    {
        if (cacheidx < i)
        {
            cache[cacheidx] = af(cache[cacheidx], cache[cacheidx + i]);
        }
        __syncthreads();
    }

    if (cacheidx == 0)
    {
        dst[blockidx] = cache[0];
    }
}

/* ----- API helper functions ----- */

const unsigned int MAX_BLOCKS = 256;

double runReduceOp(const double *src, size_t n, reducerFunc af, double identity)
{
    size_t n_src;
    size_t n_dst;
    LaunchParams lps = launchParams(n);

    double res;
    double *dst;
    double *dev_dst;

    n_src = n;
    n_dst = min(MAX_BLOCKS, lps.blockSize);
    lps.blockSize = n_dst;

    dst = (double *)(malloc(n_dst * sizeof(double)));
    handleCudaError(
        cudaMalloc(&dev_dst, n_dst * sizeof(double)));

    reduceByAssociativeFunc<<<lps.blockSize, lps.threadSize>>>(dev_dst, src, n_src, af, identity);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());
    handleCudaError(
        cudaMemcpy(
            dst,
            dev_dst,
            n_dst * sizeof(double),
            cudaMemcpyDeviceToHost));

    res = identity;
    for (size_t i = 0; i < n_dst; i++)
    {
        res = af(res, dst[i]);
    }

    free(dst);
    handleCudaError(
        cudaFree(dev_dst));

    return res;
}

/* ----- API functions ----- */

extern "C"
{
    double Sum(const double *src, size_t n);
    double Max(const double *src, size_t n);
    double Min(const double *src, size_t n);
}

double Sum(const double *src, size_t n)
{
    return runReduceOp(src, n, sum_, 0.);
}

double Max(const double *src, size_t n)
{
    return runReduceOp(src, n, max_, -INFINITY);
}

double Min(const double *src, size_t n)
{
    return runReduceOp(src, n, min_, INFINITY);
}