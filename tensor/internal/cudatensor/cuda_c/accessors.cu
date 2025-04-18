#include "types.h"
#include "common.h"

/* ----- helper functions ----- */

int elemcnt(DimArr dims)
{
    int count = 1;
    for (size_t i = 0; i < dims.size; i++)
    {
        count *= dims.arr[i];
    }

    return count;
}

int elemcnt(RangeArr ranges)
{
    int count = 1;
    for (size_t i = 0; i < ranges.size; i++)
    {
        Range range = ranges.arr[i];
        count *= range.to - range.from;
    }

    return count;
}

DimArr rcumprod(DimArr dims)
{
    DimArr rcp;

    int prod = 1;
    for (size_t i = 0; i < dims.size; i++)
    {
        size_t j = dims.size - i - 1;
        rcp.arr[j] = prod;
        prod *= dims.arr[j];
    }

    rcp.size = dims.size;

    return rcp;
}

DimArr rcumprod(RangeArr ranges)
{
    DimArr rcp;

    int prod = 1;
    for (size_t i = 0; i < ranges.size; i++)
    {
        size_t j = ranges.size - i - 1;
        Range range = ranges.arr[j];
        rcp.arr[j] = prod;
        prod *= range.to - range.from;
    }

    rcp.size = ranges.size;

    return rcp;
}

/* ----- indexing functions ----- */

__host__ __device__ int encode(DimArr index, DimArr rcp)
{
    int lnpos = 0;
    for (size_t i = 0; i < rcp.size; i++)
    {
        lnpos += index.arr[i] * rcp.arr[i];
    }

    return lnpos;
}

__host__ __device__ DimArr decode(int lnpos, DimArr rcp)
{
    DimArr index;

    int rem = lnpos;
    for (size_t i = 0; i < rcp.size; i++)
    {
        int count = rcp.arr[i];
        index.arr[i] = rem / count;
        rem = rem % count;
    }

    index.size = rcp.size;

    return index;
}

__host__ __device__ bool fallsin(DimArr index, RangeArr ranges)
{
    for (size_t i = 0; i < index.size; i++)
    {
        Range range = ranges.arr[i];
        int idx = index.arr[i];
        int from = range.from;
        int to = range.to;

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

__device__ int toSlicePosition(int lnpos_src, DimArr rcp_src, DimArr rcp_dst, RangeArr ranges)
{
    int lnpos_dst;
    DimArr index_src;
    DimArr index_dst;

    index_src = decode(lnpos_src, rcp_src);

    if (!fallsin(index_src, ranges))
    {
        return -1;
    }

    index_dst.size = index_src.size;
    for (size_t i = 0; i < index_dst.size; i++)
    {
        index_dst.arr[i] = index_src.arr[i] - ranges.arr[i].from;
    }

    lnpos_dst = encode(index_dst, rcp_dst);

    return lnpos_dst;
}

__device__ int toPatchPosition(int lnpos_src, DimArr rcp_src, DimArr rcp_dst, RangeArr ranges)
{
    int lnpos_dst;
    DimArr index_src;
    DimArr index_dst;

    index_src = decode(lnpos_src, rcp_src);

    index_dst.size = index_src.size;
    for (size_t i = 0; i < index_dst.size; i++)
    {
        index_dst.arr[i] = index_src.arr[i] + ranges.arr[i].from;
    }

    lnpos_dst = encode(index_dst, rcp_dst);

    return lnpos_dst;
}

__global__ void copySlice(
    CudaData dst,
    CudaData src,
    DimArr rcp_src,
    DimArr rcp_dst,
    RangeArr ranges)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < src.size; i += stride)
    {
        int lnpos_src = i;
        int lnpos_dst = toSlicePosition(lnpos_src, rcp_src, rcp_dst, ranges);

        if (lnpos_dst >= 0)
        {
            dst.arr[lnpos_dst] = src.arr[lnpos_src];
        }
    }
}

__global__ void copyPatch(
    CudaData dst,
    CudaData src,
    DimArr rcp_src,
    DimArr rcp_dst,
    RangeArr ranges)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < src.size; i += stride)
    {
        int lnpos_src = i;
        int lnpos_dst = toPatchPosition(lnpos_src, rcp_src, rcp_dst, ranges);

        dst.arr[lnpos_dst] = src.arr[lnpos_src];
    }
}

/* ----- API functions ----- */

extern "C"
{
    double At(CudaData src, DimArr dims, DimArr index);
    double *Slice(CudaData src, DimArr dims, RangeArr index);
    double *Patch(CudaData bas, DimArr dims, CudaData src, RangeArr index);
}

double At(CudaData src, DimArr dims, DimArr index)
{
    DimArr rcp = rcumprod(dims);
    int lnpos = encode(index, rcp);

    double elem;
    handleCudaError(
        cudaMemcpy(
            &elem,
            &src.arr[lnpos],
            sizeof(double),
            cudaMemcpyDeviceToHost));

    return elem;
}

double *Slice(CudaData src, DimArr dims, RangeArr index)
{
    size_t n = elemcnt(index);
    DimArr rcp_src = rcumprod(dims);
    DimArr rcp_dst = rcumprod(index);

    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(src.size);

    copySlice<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_src, rcp_dst, index);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *Patch(CudaData bas, DimArr dims, CudaData src, RangeArr index)
{
    size_t n = elemcnt(dims);
    DimArr rcp_src = rcumprod(index);
    DimArr rcp_dst = rcumprod(dims);

    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));
    handleCudaError(
        cudaMemcpy(
            dst.arr,
            bas.arr,
            bas.size * sizeof(double),
            cudaMemcpyDeviceToDevice));

    LaunchParams lps = launchParams(src.size);

    copyPatch<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_src, rcp_dst, index);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}