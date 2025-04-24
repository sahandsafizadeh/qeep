#include "types.h"
#include "common.h"

/* ----- API functions ----- */

extern "C"
{
    double *Reshape(CudaData src);
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