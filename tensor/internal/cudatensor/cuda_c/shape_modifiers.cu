#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

__device__ int toTransposedPosition(int lnpos_src, DimArr rcp_src, DimArr rcp_dst)
{
    int lnpos_dst;
    DimArr index_src;
    DimArr index_dst;

    index_src = decode(lnpos_src, rcp_src);
    index_dst = index_src;

    size_t n = index_dst.size;
    index_dst.arr[n - 2] = index_src.arr[n - 1];
    index_dst.arr[n - 1] = index_src.arr[n - 2];

    lnpos_dst = encode(index_dst, rcp_dst);

    return lnpos_dst;
}

__global__ void copyTranspose(CudaData dst, CudaData src, DimArr rcp_dst, DimArr rcp_src)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < src.size; i += stride)
    {
        int lnpos_src = i;
        int lnpos_dst = toTransposedPosition(lnpos_src, rcp_src, rcp_dst);

        dst.arr[lnpos_dst] = src.arr[lnpos_src];
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Transpose(CudaData src, DimArr dims_src, DimArr dims_dst);
    double *Reshape(CudaData src);
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