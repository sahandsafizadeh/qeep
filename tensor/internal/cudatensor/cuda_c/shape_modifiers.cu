#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

__device__ int toTransposedPosition(int lnpos_dst, DimArr rcp_dst, DimArr rcp_src)
{
    int lnpos_src;
    DimArr index_dst;
    DimArr index_src;

    index_dst = decode(lnpos_dst, rcp_dst);
    index_src = index_dst;

    size_t n = index_src.size;
    index_src.arr[n - 2] = index_dst.arr[n - 1];
    index_src.arr[n - 1] = index_dst.arr[n - 2];

    lnpos_src = encode(index_src, rcp_src);

    return lnpos_src;
}

__device__ int toBroadcastedPosition(int lnpos_dst, DimArr rcp_dst, DimArr rcp_src)
{
    int lnpos_src;
    DimArr index_dst;
    DimArr index_src;

    index_dst = decode(lnpos_dst, rcp_dst);
    index_src = index_dst;
    lnpos_src = encode(index_src, rcp_src);

    return lnpos_src;
}

__global__ void copyTranspose(
    CudaData dst,
    CudaData src,
    DimArr rcp_dst,
    DimArr rcp_src)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst = i;
        int lnpos_src = toTransposedPosition(lnpos_dst, rcp_dst, rcp_src);

        dst.arr[lnpos_dst] = src.arr[lnpos_src];
    }
}

__global__ void copyBroadcast(CudaData dst, CudaData src, DimArr rcp_dst, DimArr rcp_src)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        int lnpos_dst = i;
        int lnpos_src = toBroadcastedPosition(lnpos_dst, rcp_dst, rcp_src);

        dst.arr[lnpos_dst] = src.arr[lnpos_src];
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Reshape(CudaData src);
    double *Transpose(CudaData src, DimArr dims_src, DimArr dims_dst);
    double *Broadcast(CudaData src, DimArr dims_src, DimArr dims_dst);
}

double *Reshape(CudaData src)
{
    CudaData dst = (CudaData){NULL, src.size};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));
    handleCudaError(
        cudaMemcpy(
            dst.arr,
            src.arr,
            src.size * sizeof(double),
            cudaMemcpyDeviceToDevice));

    return dst.arr;
}

double *Transpose(CudaData src, DimArr dims_src, DimArr dims_dst)
{
    DimArr rcp_dst = rcumprod(dims_dst);
    DimArr rcp_src = rcumprod(dims_src);

    CudaData dst = (CudaData){NULL, src.size};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(dst.size);

    copyTranspose<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}

double *Broadcast(CudaData src, DimArr dims_src, DimArr dims_dst)
{
    size_t n = elemcnt(dims_dst);
    DimArr rcp_dst = rcumprod(dims_dst);
    DimArr rcp_src = rcumprod(dims_src);

    CudaData dst = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&dst.arr, dst.size * sizeof(double)));

    LaunchParams lps = launchParams(dst.size);

    copyBroadcast<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}