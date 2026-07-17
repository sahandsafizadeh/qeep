#include <math.h>

#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

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

/* ----- reducer implementations ----- */

struct sumReducer
{
    double value;

    __host__ __device__ void init()
    {
        value = 0.;
    }
    __host__ __device__ void feed(int _, double v)
    {
        value += v;
    }
    __host__ __device__ void merge(const sumReducer o)
    {
        value += o.value;
    }
    __host__ __device__ double result()
    {
        return value;
    }
};

struct maxReducer
{
    double value;

    __host__ __device__ void init()
    {
        value = -INFINITY;
    }
    __host__ __device__ void feed(int _, double v)
    {
        if (v > value)
        {
            value = v;
        }
    }
    __host__ __device__ void merge(const maxReducer o)
    {
        if (o.value > value)
        {
            value = o.value;
        }
    }
    __host__ __device__ double result()
    {
        return value;
    }
};

struct minReducer
{
    double value;

    __host__ __device__ void init()
    {
        value = INFINITY;
    }
    __host__ __device__ void feed(int _, double v)
    {
        if (v < value)
        {
            value = v;
        }
    }
    __host__ __device__ void merge(const minReducer o)
    {
        if (o.value < value)
        {
            value = o.value;
        }
    }
    __host__ __device__ double result()
    {
        return value;
    }
};

struct argmaxReducer
{
    int index;
    double value;

    __host__ __device__ void init()
    {
        index = -1;
        value = -INFINITY;
    }
    __host__ __device__ void feed(int i, double v)
    {
        if (v > value)
        {
            index = i;
            value = v;
        }
    }
    __host__ __device__ void merge(const argmaxReducer o)
    {
        if (o.value > value)
        {
            index = o.index;
            value = o.value;
        }
    }
    __host__ __device__ double result()
    {
        return index;
    }
};

struct argminReducer
{
    int index;
    double value;

    __host__ __device__ void init()
    {
        index = -1;
        value = INFINITY;
    }
    __host__ __device__ void feed(int i, double v)
    {
        if (v < value)
        {
            index = i;
            value = v;
        }
    }
    __host__ __device__ void merge(const argminReducer o)
    {
        if (o.value < value)
        {
            index = o.index;
            value = o.value;
        }
    }
    __host__ __device__ double result()
    {
        return index;
    }
};

struct avgReducer // Welford
{
    int count;
    double mean;

    __host__ __device__ void init()
    {
        count = 0;
        mean = 0.;
    }
    __host__ __device__ void feed(int _, double v)
    {
        count++;
        double delta = v - mean;
        mean += delta / count;
    }
    __host__ __device__ void merge(const avgReducer o)
    {
        int na = count;
        int nb = o.count;
        int n = na + nb;
        if (n == 0)
        {
            return;
        }

        double delta = o.mean - mean;

        count += nb;
        mean += delta * nb / n;
    }
    __host__ __device__ double result()
    {
        return mean;
    }
};

struct varReducer // Welford
{
    int count;
    double mean;
    double m2;

    __host__ __device__ void init()
    {
        count = 0;
        mean = 0;
        m2 = 0;
    }
    __host__ __device__ void feed(int _, double v)
    {
        count++;
        double delta = v - mean;
        mean += delta / count;
        double delta2 = v - mean;
        m2 += delta * delta2;
    }
    __host__ __device__ void merge(const varReducer o)
    {
        int na = count;
        int nb = o.count;
        int n = na + nb;
        if (n == 0)
        {
            return;
        }

        double delta = o.mean - mean;
        double delta2 = delta * delta;

        count += nb;
        mean += delta * nb / n;
        m2 += o.m2 + (delta2 * na * nb / n);
    }
    __host__ __device__ double result()
    {
        return count <= 1 ? 0. : m2 / (count - 1);
    }
};

struct stdReducer : varReducer
{
    __host__ __device__ double result()
    {
        return sqrt(varReducer::result());
    }
};

/* ----- device functions ----- */

template <typename Reducer>
__global__ void reduceAll(Reducer *o, CUDATensor t)
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

const unsigned int MAX_BLOCKS = 512;

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

    reduceDim<Reducer><<<lps.blockSize, lps.threadSize>>>(dst, t, view_o, dim);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

/* ----- API functions ----- */

extern "C"
{
    double Sum(CUDATensor t);
    double Max(CUDATensor t);
    double Min(CUDATensor t);
    double Var(CUDATensor t);
    double *Argmax(CUDATensor t, int dim, CUDAView view_o);
    double *Argmin(CUDATensor t, int dim, CUDAView view_o);
    double *SumAlong(CUDATensor t, int dim, CUDAView view_o);
    double *MaxAlong(CUDATensor t, int dim, CUDAView view_o);
    double *MinAlong(CUDATensor t, int dim, CUDAView view_o);
    double *AvgAlong(CUDATensor t, int dim, CUDAView view_o);
    double *VarAlong(CUDATensor t, int dim, CUDAView view_o);
    double *StdAlong(CUDATensor t, int dim, CUDAView view_o);
}

double Sum(CUDATensor t)
{
    return runAllReducer<sumReducer>(t);
}

double Max(CUDATensor t)
{
    return runAllReducer<maxReducer>(t);
}

double Min(CUDATensor t)
{
    return runAllReducer<minReducer>(t);
}

double Var(CUDATensor t)
{
    return runAllReducer<varReducer>(t);
}

double *Argmax(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<argmaxReducer>(t, dim, view_o);
}

double *Argmin(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<argminReducer>(t, dim, view_o);
}

double *SumAlong(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<sumReducer>(t, dim, view_o);
}

double *MaxAlong(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<maxReducer>(t, dim, view_o);
}

double *MinAlong(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<minReducer>(t, dim, view_o);
}

double *AvgAlong(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<avgReducer>(t, dim, view_o);
}

double *VarAlong(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<varReducer>(t, dim, view_o);
}

double *StdAlong(CUDATensor t, int dim, CUDAView view_o)
{
    return runDimReducer<stdReducer>(t, dim, view_o);
}