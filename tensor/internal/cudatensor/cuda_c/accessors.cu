#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

__device__ bool fallsin(DimArr index, RangeArr ranges)
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
    DimArr rcp_dst,
    DimArr rcp_src,
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
    DimArr rcp_dst,
    DimArr rcp_src,
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
    DimArr rcp_dst = rcumprod(index);
    DimArr rcp_src = rcumprod(dims);

    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(src.size);

    copySlice<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, index);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *Patch(CudaData bas, DimArr dims, CudaData src, RangeArr index)
{
    size_t n = elemcnt(dims);
    DimArr rcp_dst = rcumprod(dims);
    DimArr rcp_src = rcumprod(index);

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

    copyPatch<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, index);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}