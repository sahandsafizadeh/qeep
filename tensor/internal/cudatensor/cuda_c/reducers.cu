#include <math.h>

#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- reducer implementations ----- */

struct sumReducer
{
    double value;

    __host__ __device__ void init()
    {
        value = 0.;
    }
    __host__ __device__ void feed(size_t _, double v)
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
    __host__ __device__ void feed(size_t _, double v)
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
    __host__ __device__ void feed(size_t _, double v)
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
    size_t index;
    double value;

    __host__ __device__ void init()
    {
        index = 0;
        value = -INFINITY;
    }
    __host__ __device__ void feed(size_t i, double v)
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
    size_t index;
    double value;

    __host__ __device__ void init()
    {
        index = 0;
        value = INFINITY;
    }
    __host__ __device__ void feed(size_t i, double v)
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
    size_t count;
    double mean;

    __host__ __device__ void init()
    {
        count = 0;
        mean = 0.;
    }
    __host__ __device__ void feed(size_t _, double v)
    {
        count++;
        double delta = v - mean;
        mean += delta / count;
    }
    __host__ __device__ void merge(const avgReducer o)
    {
        size_t na = count;
        size_t nb = o.count;
        size_t n = na + nb;
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
    size_t count;
    double mean;
    double m2;

    __host__ __device__ void init()
    {
        count = 0;
        mean = 0;
        m2 = 0;
    }
    __host__ __device__ void feed(size_t _, double v)
    {
        count++;
        double delta = v - mean;
        mean += delta / count;
        double delta2 = v - mean;
        m2 += delta * delta2;
    }
    __host__ __device__ void merge(const varReducer o)
    {
        size_t na = count;
        size_t nb = o.count;
        size_t n = na + nb;
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

__device__ size_t flatpos(size_t i, CUDAView view)
{
    DimArr index;
    index.size = view.dims.size;

    size_t rem = i;
    for (int j = index.size - 1; j >= 0; j--)
    {
        size_t dim = view.dims.arr[j];
        index.arr[j] = rem % dim;
        rem /= dim;
    }

    return index2lnpos(index, view);
}

template <typename Reducer>
__global__ void reduceAll(Reducer *o, CUDATensor t)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    Reducer r;

    r.init();
    for (size_t i = tpos; i < t.data.size; i += stride)
    {
        size_t lnpos_t = flatpos(i, t.view);
        r.feed(i, t.data.arr[lnpos_t]);
    }

    const unsigned int cacheidx = threadIndex();
    const unsigned int cachelen = blockSize();
    const unsigned int blockidx = blockIndex();

    __shared__ Reducer cache[MAX_THREADS_PER_BLOCK_X];

    cache[cacheidx] = r;
    __syncthreads();

    for (size_t i = cachelen / 2; i != 0; i /= 2)
    {
        if (cacheidx < i)
        {
            cache[cacheidx].merge(cache[cacheidx + i]);
        }
        __syncthreads();
    }

    if (cacheidx == 0)
    {
        o[blockidx] = cache[0];
    }
}

template <typename Reducer>
__global__ void reduceDim(CUDATensor o, CUDATensor t, int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        DimArr index_o = lnpos2index(i, o.view);
        DimArr index_t = index_o;
        index_t.size = index_o.size + 1;
        for (int d = index_o.size; d > dim; d--)
        {
            index_t.arr[d] = index_t.arr[d - 1];
        }

        Reducer r;
        size_t cdim = t.view.dims.arr[dim];

        r.init();
        for (size_t j = 0; j < cdim; j++)
        {
            index_t.arr[dim] = j;
            size_t lnpos_t = index2lnpos(index_t, t.view);

            r.feed(j, t.data.arr[lnpos_t]);
        }

        o.data.arr[i] = r.result();
    }
}

/* ----- API helper functions ----- */

const unsigned int MAX_BLOCKS = 512;

template <typename Reducer>
double runAllReducer(CUDATensor t)
{
    size_t n = elemcnt(t.view.dims);
    LaunchParams lps = launchParams(n);

    n = min(lps.blockSize, MAX_BLOCKS);
    lps.blockSize = n;

    Reducer *dev_o;
    handleCudaError(
        cudaMalloc(&dev_o, n * sizeof(Reducer)));

    reduceAll<Reducer><<<lps.blockSize, lps.threadSize>>>(dev_o, t);
    handleCudaError(
        cudaGetLastError());

    Reducer *hst_o = (Reducer *)(malloc(n * sizeof(Reducer)));
    handleCudaError(
        cudaMemcpy(
            hst_o,
            dev_o,
            n * sizeof(Reducer),
            cudaMemcpyDeviceToHost));
    handleCudaError(
        cudaFree(dev_o));

    Reducer acc;
    acc.init();
    for (size_t i = 0; i < n; i++)
    {
        acc.merge(hst_o[i]);
    }

    double res = acc.result();
    free(hst_o);

    return res;
}

template <typename Reducer>
double *runDimReducer(CUDATensor t, int dim, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    reduceDim<Reducer><<<lps.blockSize, lps.threadSize>>>(o, t, dim);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

/* ----- API functions ----- */

extern "C"
{
    double Sum(CUDATensor t);
    double Max(CUDATensor t);
    double Min(CUDATensor t);
    double Avg(CUDATensor t);
    double Var(CUDATensor t);
    double Std(CUDATensor t);
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

double Avg(CUDATensor t)
{
    return runAllReducer<avgReducer>(t);
}

double Var(CUDATensor t)
{
    return runAllReducer<varReducer>(t);
}

double Std(CUDATensor t)
{
    return runAllReducer<stdReducer>(t);
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