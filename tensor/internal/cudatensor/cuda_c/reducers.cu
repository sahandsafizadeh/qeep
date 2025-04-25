#include <math.h>

#include "common.cuh"

/* ----- device functions ----- */

enum ReduceType
{
    RED_SUM,
    RED_MAX,
    RED_MIN,
};

__host__ __device__ double reduce(double a, double b, ReduceType rdt)
{
    switch (rdt)
    {
    case RED_SUM:
        return a + b;
    case RED_MAX:
        return a >= b ? a : b;
    case RED_MIN:
        return a <= b ? a : b;
    }

    return NAN;
}

__global__ void reduceByAssociativeFunc(
    double *dst,
    const double *src,
    size_t n,
    ReduceType rdt,
    double identity)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    double temp = identity;
    for (size_t i = tpos; i < n; i += stride)
    {
        temp = reduce(temp, src[i], rdt);
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
            cache[cacheidx] = reduce(cache[cacheidx], cache[cacheidx + i], rdt);
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

double runReduceOp(const double *src, size_t n, ReduceType rdt, double identity)
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

    reduceByAssociativeFunc<<<lps.blockSize, lps.threadSize>>>(dev_dst, src, n_src, rdt, identity);

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
        res = reduce(res, dst[i], rdt);
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
    return runReduceOp(src, n, RED_SUM, 0.);
}

double Max(const double *src, size_t n)
{
    return runReduceOp(src, n, RED_MAX, -INFINITY);
}

double Min(const double *src, size_t n)
{
    return runReduceOp(src, n, RED_MIN, INFINITY);
}