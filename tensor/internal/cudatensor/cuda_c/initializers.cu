#include <time.h>
#include <curand_kernel.h>

#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- helper functions ----- */

inline unsigned long long timeSeed()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
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

__global__ void fillEye(CUDATensor o, size_t d)
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
        DimArr index_o = lnpos2index(i, o.view);
        int lnpos_t = index2lnpos(index_o, t.view);

        o.data.arr[i] = t.data.arr[lnpos_t];
    }
}

__global__ void fillConcat(CUDATensor o, CUDATensor *ts, size_t *ofsts, int size, int dim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        DimArr index_o = lnpos2index(i, o.view);

        int s = 0;
        while (s < size - 1 && index_o.arr[dim] >= ofsts[s + 1])
        {
            s++;
        }

        DimArr index_t = index_o;
        index_t.arr[dim] -= ofsts[s];
        size_t lnpos_t = index2lnpos(index_t, ts[s].view);

        o.data.arr[i] = ts[s].data.arr[lnpos_t];
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
    double *Concat(CUDATensor ts[], int size, int dim, CUDAView view_o);
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
    size_t d = view_o.dims.arr[0];
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

double *Concat(CUDATensor ts[], int size, int dim, CUDAView view_o)
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

    size_t *ofsts = (size_t *)malloc((size + 1) * sizeof(size_t));
    ofsts[0] = 0;
    for (int i = 0; i < size; i++)
    {
        ofsts[i + 1] = ofsts[i] + ts[i].view.dims.arr[dim];
    }

    size_t *_ofsts;
    handleCudaError(
        cudaMalloc(&_ofsts, size * sizeof(size_t)));
    handleCudaError(
        cudaMemcpy(
            _ofsts,
            ofsts,
            size * sizeof(size_t),
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