#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

__global__ void compact(CUDATensor o, CUDATensor t)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        size_t lnpos_t = flatpos(i, t.view);
        o.data.arr[i] = t.data.arr[lnpos_t];
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Compact(CUDATensor t, CUDAView view_o);
}

double *Compact(CUDATensor t, CUDAView view_o)
{
    CUDAData data_o = (CUDAData){t.data.size, NULL};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    compact<<<lps.blockSize, lps.threadSize>>>(o, t);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}
