#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- device functions ----- */

__device__ bool fallsin(DimArr index, RangeArr ranges)
{
    for (int i = 0; i < index.size; i++)
    {
        Range range = ranges.arr[i];
        size_t idx = index.arr[i];
        size_t from = range.from;
        size_t to = range.to;

        if (idx < from || idx >= to)
        {
            return false;
        }
    }

    return true;
}

__device__ size_t patchpos(DimArr index, RangeArr ranges, CUDAView view)
{
    for (int i = 0; i < index.size; i++)
    {
        index.arr[i] -= ranges.arr[i].from;
    }

    return index2lnpos(index, view);
}

__global__ void applyPatch(CUDATensor o, CUDATensor t, RangeArr ranges, CUDATensor u)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < o.data.size; i += stride)
    {
        double value;

        DimArr index_o = lnpos2index(i, o.view);
        if (!fallsin(index_o, ranges))
        {
            size_t lnpos_t = index2lnpos(index_o, t.view);
            value = t.data.arr[lnpos_t];
        }
        else
        {
            size_t lnpos_t = patchpos(index_o, ranges, u.view);
            value = u.data.arr[lnpos_t];
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
    size_t lnpos = index2lnpos(index, t.view);

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

    CUDAData data_o = (CUDAData){n, NULL};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyPatch<<<lps.blockSize, lps.threadSize>>>(o, t, ranges, u);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}