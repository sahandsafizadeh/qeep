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

        if (idx < from || idx >= to)
        {
            return false;
        }
    }

    return true;
}

__device__ int toPatchPosition(DimArr index, RangeArr ranges, CUDAView view)
{
    for (size_t i = 0; i < index.size; i++)
    {
        index.arr[i] -= ranges.arr[i].from;
    }

    return index2lnpos(index, view);
}

__global__ void applyPatch(CUDATensor o, CUDATensor t, CUDATensor u, RangeArr ranges)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        double value;

        DimArr oidx = lnpos2index(i, o.view);
        if (!fallsin(oidx, ranges))
        {
            int lnpos_src = index2lnpos(oidx, t.view);
            value = t.data.arr[lnpos_src];
        }
        else
        {
            int lnpos_src = toPatchPosition(oidx, ranges, u.view);
            value = u.data.arr[lnpos_src];
        }

        o.data.arr[i] = value;
    }
}

/* ----- API functions ----- */

extern "C"
{
    double At(CUDATensor t, DimArr index);
    double *Patch(CUDATensor t, RangeArr ranges, CUDATensor u, CUDAView view_o);
}

double At(CUDATensor t, DimArr index)
{
    int lnpos = index2lnpos(index, t.view);

    double elem;
    handleCudaError(
        cudaMemcpy(
            &elem,
            &t.data.arr[lnpos],
            sizeof(double),
            cudaMemcpyDeviceToHost));

    return elem;
}

double *Patch(CUDATensor t, RangeArr ranges, CUDATensor u, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyPatch<<<lps.blockSize, lps.threadSize>>>(o, t, u, ranges);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}