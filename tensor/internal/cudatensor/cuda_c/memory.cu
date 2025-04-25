#include "common.cuh"

extern "C"
{
    void GetCudaMemInfo(size_t *free_mem, size_t *total_mem);
    void FreeCudaMem(double *data);
}

void GetCudaMemInfo(size_t *free_mem, size_t *total_mem)
{
    handleCudaError(
        cudaMemGetInfo(free_mem, total_mem));
}

void FreeCudaMem(double *data)
{
    handleCudaError(
        cudaFree(data));
}