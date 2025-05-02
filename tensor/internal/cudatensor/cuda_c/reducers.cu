#include <math.h>

#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

enum ReduceType
{
    RED_SUM,
    RED_MAX,
    RED_MIN,
    RED_AVG,
    RED_VAR,
    RED_STD,
    RED_ARGMAX,
    RED_ARGMIN,
};

__device__ DimArr unsqueezeidx(DimArr index_dst, int dim)
{
    DimArr index_src;

    index_src.size = index_dst.size + 1;
    for (size_t i = 0; i < index_src.size; i++)
    {
        if (i < dim)
        {
            index_src.arr[i] = index_dst.arr[i];
        }
        else if (i == dim)
        {
            index_src.arr[i] = 0;
        }
        else
        {
            index_src.arr[i] = index_dst.arr[i - 1];
        }
    }

    return index_src;
}

__global__ void reduceDimByArgmax(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double tempidx = -1;
        double tempval = -INFINITY;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            if (src.arr[lnpos_src] > tempval)
            {
                tempidx = i;
                tempval = src.arr[lnpos_src];
            }
        }

        dst.arr[lnpos_dst] = tempidx;
    }
}

__global__ void reduceDimByArgmin(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double tempidx = -1;
        double tempval = INFINITY;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            if (src.arr[lnpos_src] < tempval)
            {
                tempidx = i;
                tempval = src.arr[lnpos_src];
            }
        }

        dst.arr[lnpos_dst] = tempidx;
    }
}

__global__ void reduceDimBySum(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double temp = 0.;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            temp = temp + src.arr[lnpos_src];
        }

        dst.arr[lnpos_dst] = temp;
    }
}

__global__ void reduceDimByMax(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double temp = -INFINITY;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            if (src.arr[lnpos_src] > temp)
            {
                temp = src.arr[lnpos_src];
            }
        }

        dst.arr[lnpos_dst] = temp;
    }
}

__global__ void reduceDimByMin(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double temp = INFINITY;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            if (src.arr[lnpos_src] < temp)
            {
                temp = src.arr[lnpos_src];
            }
        }

        dst.arr[lnpos_dst] = temp;
    }
}

__global__ void reduceDimByAvg(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double temp = 0.;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            temp = temp + src.arr[lnpos_src];
        }

        dst.arr[lnpos_dst] = temp / diml;
    }
}

__global__ void reduceDimByVar(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double average = 0.;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            average = average + src.arr[lnpos_src];
        }

        average /= diml;

        double temp = 0.;
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            temp = temp + pow(src.arr[lnpos_src] - average, 2);
        }

        if (diml > 1)
        {
            dst.arr[lnpos_dst] = temp / (diml - 1);
        }
        else
        {
            dst.arr[lnpos_dst] = 0.;
        }
    }
}

__global__ void reduceDimByStd(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src,
    DimArr dims,
    int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = unsqueezeidx(index_dst, dim);

        double average = 0.;
        double diml = dims.arr[dim];
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            average = average + src.arr[lnpos_src];
        }

        average /= diml;

        double temp = 0.;
        for (size_t i = 0; i < diml; i++)
        {
            index_src.arr[dim] = i;
            lnpos_src = encode(index_src, rcp_src);
            temp = temp + pow(src.arr[lnpos_src] - average, 2);
        }

        if (diml > 1)
        {
            dst.arr[lnpos_dst] = sqrt(temp / (diml - 1));
        }
        else
        {
            dst.arr[lnpos_dst] = 0.;
        }
    }
}

__host__ __device__ inline double reduce(double a, double b, ReduceType rdt)
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

double *runDimReducer(
    CudaData src,
    int dim,
    DimArr dims_src,
    DimArr dims_dst,
    ReduceType rdt)
{
    size_t n = elemcnt(dims_dst);
    DimArr rcp_dst = rcumprod(dims_dst);
    DimArr rcp_src = rcumprod(dims_src);

    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(dst.size);

    switch (rdt)
    {
    case RED_SUM:
        reduceDimBySum<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_MAX:
        reduceDimByMax<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_MIN:
        reduceDimByMin<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_AVG:
        reduceDimByAvg<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_VAR:
        reduceDimByVar<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_STD:
        reduceDimByStd<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_ARGMAX:
        reduceDimByArgmax<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    case RED_ARGMIN:
        reduceDimByArgmin<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, dims_src, dim);
        break;
    }

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

/* ----- API functions ----- */

extern "C"
{
    double Sum(const double *src, size_t n);
    double Max(const double *src, size_t n);
    double Min(const double *src, size_t n);
    double *Argmax(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *Argmin(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *SumAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *MaxAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *MinAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *AvgAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *VarAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *StdAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
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

double *Argmax(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_ARGMAX);
}

double *Argmin(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_ARGMIN);
}

double *SumAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_SUM);
}

double *MaxAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_MAX);
}

double *MinAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_MIN);
}

double *AvgAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_AVG);
}

double *VarAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_VAR);
}

double *StdAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst)
{
    return runDimReducer(src, dim, dims_src, dims_dst, RED_STD);
}