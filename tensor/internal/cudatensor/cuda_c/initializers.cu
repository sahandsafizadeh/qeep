#include <time.h>
#include <curand_kernel.h>

#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- helper functions ----- */

inline unsigned long long timeSeed()
{
    return (unsigned long long)time(NULL);
}

/* ----- device functions ----- */

__global__ void fillConst(CUDATensor o, double value)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        o.data.arr[i] = value;
    }
}

__global__ void fillEye(CUDATensor o, int d)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        o.data.arr[i] = i % (d + 1) == 0 ? 1. : 0.;
    }
}

__global__ void fillRandU(CUDATensor o, double l, double u, unsigned long long seed)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tpos, 0, &state);

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        double randu_0_1 = curand_uniform_double(&state);
        o.data.arr[i] = l + (u - l) * randu_0_1;
    }
}

__global__ void fillRandN(CUDATensor o, double u, double s, unsigned long long seed)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tpos, 0, &state);

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        double randn_0_1 = curand_normal_double(&state);
        o.data.arr[i] = u + s * randn_0_1;
    }
}

__global__ void fillCopy(CUDATensor o, CUDATensor t)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        DimArr index_dst = lnpos2index(i, o.view);
        int lnpos_src = index2lnpos(index_dst, t.view);

        o.data.arr[i] = t.data.arr[lnpos_src];
    }
}

__global__ void fillConcat(CUDATensor o, CUDATensor *ts, int *ofsts, size_t size, int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        DimArr index_dst = lnpos2index(i, o.view);

        size_t s = 0;
        while (s < size - 1 && index_dst.arr[dim] >= ofsts[s + 1])
        {
            s++;
        }

        DimArr index_src = index_dst;
        index_src.arr[dim] -= ofsts[s];
        int lnpos_src = index2lnpos(index_src, ts[s].view);

        o.data.arr[i] = ts[s].data.arr[lnpos_src];
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Full(double value, CUDAView view_o);
    double *Eye(CUDAView view_o);
    double *RandU(double l, double u, CUDAView view_o);
    double *RandN(double u, double s, CUDAView view_o);
    double *Of(double *input_data, CUDAView view_o);
    double *From(CUDATensor t, CUDAView view_o);
    double *Concat(CUDATensor ts[], size_t size, int dim, CUDAView view_o);
}

double *Full(double value, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    fillConst<<<lps.blockSize, lps.threadSize>>>(o, value);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *Eye(CUDAView view_o)
{
    int d = view_o.dims.arr[0];
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    fillEye<<<lps.blockSize, lps.threadSize>>>(o, d);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *RandU(double l, double u, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    unsigned long long seed = timeSeed();

    LaunchParams lps = launchParams(o.data.size);
    fillRandU<<<lps.blockSize, lps.threadSize>>>(o, l, u, seed);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *RandN(double u, double s, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    unsigned long long seed = timeSeed();

    LaunchParams lps = launchParams(o.data.size);
    fillRandN<<<lps.blockSize, lps.threadSize>>>(o, u, s, seed);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *Of(double *input_data, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    handleCudaError(
        cudaMemcpy(
            data_o.arr,
            input_data,
            data_o.size * sizeof(double),
            cudaMemcpyHostToDevice));

    return data_o.arr;
}

double *From(CUDATensor t, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    fillCopy<<<lps.blockSize, lps.threadSize>>>(o, t);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *Concat(CUDATensor ts[], size_t size, int dim, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor *_ts;
    handleCudaError(
        cudaMalloc(&_ts, size * sizeof(CUDATensor)));
    handleCudaError(
        cudaMemcpy(
            _ts,
            ts,
            size * sizeof(CUDATensor),
            cudaMemcpyHostToDevice));

    int *ofsts = (int *)malloc((size + 1) * sizeof(int));
    ofsts[0] = 0;
    for (size_t i = 0; i < size; i++)
    {
        DimArr tdims = ts[i].view.dims;
        ofsts[i + 1] = ofsts[i] + tdims.arr[dim];
    }

    int *_ofsts;
    handleCudaError(
        cudaMalloc(&_ofsts, size * sizeof(int)));
    handleCudaError(
        cudaMemcpy(
            _ofsts,
            ofsts,
            size * sizeof(int),
            cudaMemcpyHostToDevice));
    free(ofsts);

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    fillConcat<<<lps.blockSize, lps.threadSize>>>(o, _ts, _ofsts, size, dim);
    handleCudaError(
        cudaGetLastError());

    cudaFree(_ts);
    cudaFree(_ofsts);

    return o.data.arr;
}