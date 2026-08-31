#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

/* ----- API functions ----- */

extern "C"
{
    double At(CUDATensor t, DimArr index);
    void Export(CUDATensor t, double *output_data, CUDAView view_o);
    double *Compact(CUDATensor t, CUDAView view_o); // used from shape modifiers
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

void Export(CUDATensor t, double *output_data, CUDAView view_o)
{
    double *compact_data = Compact(t, view_o);

    handleCudaError(
        cudaMemcpy(
            output_data,
            compact_data,
            t.data.size * sizeof(double),
            cudaMemcpyDeviceToHost));

    handleCudaError(
        cudaFree(compact_data));
}