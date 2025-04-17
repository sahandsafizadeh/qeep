#include "types.h"
#include "common.h"

/* ----- helper functions ----- */

int elemcnt(const int *dims, size_t n)
{
    int count = 1;
    for (size_t i = 0; i < n; i++)
    {
        count *= dims[i];
    }

    return count;
}

int elemcnt(const Range *range, size_t n)
{
    int count = 1;
    for (size_t i = 0; i < n; i++)
    {
        count *= range[i].to - range[i].from;
    }

    return count;
}

/* ----- indexing functions ----- */

void rcumprod(int *rcp, const int *dims, size_t n)
{
    int prod = 1;
    for (size_t i = 0; i < n; i++)
    {
        size_t j = n - i - 1;
        rcp[j] = prod;
        prod *= dims[j];
    }
}

void rcumprod(int *rcp, const Range *range, size_t n)
{
    int prod = 1;
    for (size_t i = 0; i < n; i++)
    {
        size_t j = n - i - 1;
        rcp[j] = prod;
        prod *= range[j].to - range[j].from;
    }
}

__host__ __device__ void encode(int *lnpos, const int *index, const int *rcp, size_t n)
{
    *lnpos = 0;
    for (size_t i = 0; i < n; i++)
    {
        size_t j = n - i - 1;
        *lnpos += index[j] * rcp[j];
    }
}

__host__ __device__ void decode(int *index, int lnpos, const int *rcp, size_t n)
{
    int rem = lnpos;
    for (size_t i = 0; i < n; i++)
    {
        index[i] = rem / rcp[i];
        rem %= rcp[i];
    }
}

__host__ __device__ bool fallsin(const int *index, const Range *range, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        int idx = index[i];
        int from = range[i].from;
        int to = range[i].to;

        if (!(from <= idx && idx < to))
        {
            return false;
        }
    }

    return true;
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

__device__ int toSlicePosition(
    int lnpos_src,
    const int *rcp_src,
    const int *rcp_dst,
    const Range *range,
    size_t n)
{
    int lnpos_dst;
    int *index_src = (int *)(malloc(n * sizeof(int)));
    int *index_dst = (int *)(malloc(n * sizeof(int)));

    decode(index_src, lnpos_src, rcp_src, n);

    if (!fallsin(index_src, range, n))
    {
        return -1;
    }

    for (size_t i = 0; i < n; i++)
    {
        index_dst[i] = index_src[i] - range[i].from;
    }

    encode(&lnpos_dst, index_dst, rcp_dst, n);

    free(index_src);
    free(index_dst);

    return lnpos_dst;
}

__device__ int toPatchPosition(
    int lnpos_src,
    const int *rcp_src,
    const int *rcp_dst,
    const Range *range,
    size_t n)
{
    int lnpos_dst;
    int *index_src = (int *)(malloc(n * sizeof(int)));
    int *index_dst = (int *)(malloc(n * sizeof(int)));

    decode(index_src, lnpos_src, rcp_src, n);

    for (size_t i = 0; i < n; i++)
    {
        index_dst[i] = index_src[i] + range[i].from;
    }

    encode(&lnpos_dst, index_dst, rcp_dst, n);

    free(index_src);
    free(index_dst);

    return lnpos_dst;
}

__global__ void copySlice(
    double *dst,
    const double *src,
    size_t n_src,
    const int *rcp_src,
    const int *rcp_dst,
    const Range *range,
    size_t n_index)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n_src; i += stride)
    {
        int lnpos_src = i;
        int lnpos_dst = toSlicePosition(lnpos_src, rcp_src, rcp_dst, range, n_index);

        if (lnpos_dst >= 0)
        {
            dst[lnpos_dst] = src[lnpos_src];
        }
    }
}

__global__ void copyPatch(
    double *dst,
    const double *src,
    size_t n_src,
    const int *rcp_src,
    const int *rcp_dst,
    const Range *range,
    size_t n_index)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n_src; i += stride)
    {
        int lnpos_src = i;
        int lnpos_dst = toPatchPosition(lnpos_src, rcp_src, rcp_dst, range, n_index);

        dst[lnpos_dst] = src[lnpos_src];
    }
}

/* ----- API functions ----- */

extern "C"
{
    double At(const double *data, const int *dims, const int *index, size_t n);
    double *Slice(const double *src, const int *dims, const Range *index, size_t n);
}

double At(const double *data, const int *dims, const int *index, size_t n)
{
    int lnpos;
    int *rcp = (int *)(malloc(n * sizeof(int)));

    rcumprod(rcp, dims, n);
    encode(&lnpos, index, rcp, n);
    free(rcp);

    double elem;
    handleCudaError(
        cudaMemcpy(
            &elem,
            &data[lnpos],
            sizeof(double),
            cudaMemcpyDeviceToHost));

    return elem;
}

double *Slice(const double *src, const int *dims, const Range *index, size_t n)
{
    int n_index = n;
    int n_src = elemsCount(dims, n_index);
    int n_dst = elemsCount(index, n_index);

    int *rcp_src = (int *)(malloc(n_index * sizeof(int)));
    int *rcp_dst = (int *)(malloc(n_index * sizeof(int)));
    rcumprod(rcp_src, dims, n_index);
    rcumprod(rcp_dst, index, n_index);

    double *dst;
    handleCudaError(
        cudaMalloc(&dst, n_dst * sizeof(double)));

    LaunchParams lps = launchParams(n_src);

    copySlice<<<lps.blockSize, lps.threadSize>>>(dst, src, n_src,
                                                 rcp_src, rcp_dst, index, n_index);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    free(rcp_src);
    free(rcp_dst);

    return dst;
}