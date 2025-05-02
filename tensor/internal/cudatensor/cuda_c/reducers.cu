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

__global__ void reduceBySum(CudaData dst, CudaData src)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    double temp = 0.;
    for (size_t i = tpos; i < src.size; i += stride)
    {
        temp = temp + src.arr[i];
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
            cache[cacheidx] = cache[cacheidx] + cache[cacheidx + i];
        }
        __syncthreads();
    }

    if (cacheidx == 0)
    {
        dst.arr[blockidx] = cache[0];
    }
}

__global__ void reduceByMax(CudaData dst, CudaData src)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    double temp = -INFINITY;
    for (size_t i = tpos; i < src.size; i += stride)
    {
        if (src.arr[i] > temp)
        {
            temp = src.arr[i];
        }
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
            if (cache[cacheidx + i] > cache[cacheidx])
            {
                cache[cacheidx] = cache[cacheidx + i];
            }
        }
        __syncthreads();
    }

    if (cacheidx == 0)
    {
        dst.arr[blockidx] = cache[0];
    }
}

__global__ void reduceByMin(CudaData dst, CudaData src)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    double temp = INFINITY;
    for (size_t i = tpos; i < src.size; i += stride)
    {
        if (src.arr[i] < temp)
        {
            temp = src.arr[i];
        }
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
            if (cache[cacheidx + i] < cache[cacheidx])
            {
                cache[cacheidx] = cache[cacheidx + i];
            }
        }
        __syncthreads();
    }

    if (cacheidx == 0)
    {
        dst.arr[blockidx] = cache[0];
    }
}

__global__ void reduceByVar(CudaData dst, CudaData src, double average)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    double temp = 0.;
    for (size_t i = tpos; i < src.size; i += stride)
    {
        temp = temp + pow(src.arr[i] - average, 2);
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
            cache[cacheidx] = cache[cacheidx] + cache[cacheidx + i];
        }
        __syncthreads();
    }

    if (cacheidx == 0)
    {
        dst.arr[blockidx] = cache[0];
    }
}

/* ----- API helper functions ----- */

const unsigned int MAX_BLOCKS = 256;

double runSumReducer(CudaData src)
{
    size_t n;
    CudaData dev_dst;
    double *host_dst;

    LaunchParams lps = launchParams(src.size);
    n = min(lps.blockSize, MAX_BLOCKS);
    lps.blockSize = n;

    dev_dst = (CudaData){NULL, n};
    host_dst = (double *)(malloc(n * sizeof(double)));
    handleCudaError(
        cudaMalloc(&dev_dst.arr, dev_dst.size * sizeof(double)));

    reduceBySum<<<lps.blockSize, lps.threadSize>>>(dev_dst, src);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());
    handleCudaError(
        cudaMemcpy(
            host_dst,
            dev_dst.arr,
            dev_dst.size * sizeof(double),
            cudaMemcpyDeviceToHost));

    double res = 0.;
    for (size_t i = 0; i < n; i++)
    {
        res = res + host_dst[i];
    }

    free(host_dst);
    handleCudaError(
        cudaFree(dev_dst.arr));

    return res;
}

double runMaxReducer(CudaData src)
{
    size_t n;
    CudaData dev_dst;
    double *host_dst;

    LaunchParams lps = launchParams(src.size);
    n = min(lps.blockSize, MAX_BLOCKS);
    lps.blockSize = n;

    dev_dst = (CudaData){NULL, n};
    host_dst = (double *)(malloc(n * sizeof(double)));
    handleCudaError(
        cudaMalloc(&dev_dst.arr, dev_dst.size * sizeof(double)));

    reduceByMax<<<lps.blockSize, lps.threadSize>>>(dev_dst, src);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());
    handleCudaError(
        cudaMemcpy(
            host_dst,
            dev_dst.arr,
            dev_dst.size * sizeof(double),
            cudaMemcpyDeviceToHost));

    double res = -INFINITY;
    for (size_t i = 0; i < n; i++)
    {
        if (host_dst[i] > res)
        {
            res = host_dst[i];
        }
    }

    free(host_dst);
    handleCudaError(
        cudaFree(dev_dst.arr));

    return res;
}

double runMinReducer(CudaData src)
{
    size_t n;
    CudaData dev_dst;
    double *host_dst;

    LaunchParams lps = launchParams(src.size);
    n = min(lps.blockSize, MAX_BLOCKS);
    lps.blockSize = n;

    dev_dst = (CudaData){NULL, n};
    host_dst = (double *)(malloc(n * sizeof(double)));
    handleCudaError(
        cudaMalloc(&dev_dst.arr, dev_dst.size * sizeof(double)));

    reduceByMin<<<lps.blockSize, lps.threadSize>>>(dev_dst, src);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());
    handleCudaError(
        cudaMemcpy(
            host_dst,
            dev_dst.arr,
            dev_dst.size * sizeof(double),
            cudaMemcpyDeviceToHost));

    double res = INFINITY;
    for (size_t i = 0; i < n; i++)
    {
        if (host_dst[i] < res)
        {
            res = host_dst[i];
        }
    }

    free(host_dst);
    handleCudaError(
        cudaFree(dev_dst.arr));

    return res;
}

double runVarReducer(CudaData src)
{
    size_t n;
    CudaData dev_dst;
    double *host_dst;
    double average;

    LaunchParams lps = launchParams(src.size);
    n = min(lps.blockSize, MAX_BLOCKS);
    lps.blockSize = n;

    dev_dst = (CudaData){NULL, n};
    host_dst = (double *)(malloc(n * sizeof(double)));
    handleCudaError(
        cudaMalloc(&dev_dst.arr, dev_dst.size * sizeof(double)));

    average = runSumReducer(src) / src.size;

    reduceByVar<<<lps.blockSize, lps.threadSize>>>(dev_dst, src, average);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());
    handleCudaError(
        cudaMemcpy(
            host_dst,
            dev_dst.arr,
            dev_dst.size * sizeof(double),
            cudaMemcpyDeviceToHost));

    double res = 0.;
    for (size_t i = 0; i < n; i++)
    {
        res = res + host_dst[i];
    }

    if (src.size > 1)
    {
        res = res / (src.size - 1);
    }
    else
    {
        res = 0.;
    }

    free(host_dst);
    handleCudaError(
        cudaFree(dev_dst.arr));

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
    double Sum(CudaData src);
    double Max(CudaData src);
    double Min(CudaData src);
    double Var(CudaData src);
    double *Argmax(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *Argmin(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *SumAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *MaxAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *MinAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *AvgAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *VarAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
    double *StdAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
}

double Sum(CudaData src)
{
    return runSumReducer(src);
}

double Max(CudaData src)
{
    return runMaxReducer(src);
}

double Min(CudaData src)
{
    return runMinReducer(src);
}

double Var(CudaData src)
{
    return runVarReducer(src);
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