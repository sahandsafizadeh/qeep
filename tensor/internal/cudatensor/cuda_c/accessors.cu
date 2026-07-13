#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

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

__global__ void applyPatch(
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
    double At(CUDATensor t, DimArr index);
    double *Patch(CUDATensor t, CUDATensor u, RangeArr index);
}

double At(CUDATensor t, DimArr index)
{
    int lnpos = index2lnpos(index, t.view);

    double elem;
    handleCudaError(
        cudaMemcpy(
            &elem,
            &src.arr[lnpos],
            sizeof(double),
            cudaMemcpyDeviceToHost));

    return elem;
}

double *Patch(CUDATensor t, CUDATensor u, RangeArr index)
{
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

    applyPatch<<<lps.blockSize, lps.threadSize>>>(dst, src, rcp_dst, rcp_src, index);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return dst.arr;
}