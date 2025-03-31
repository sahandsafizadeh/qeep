#include "common.h"

extern "C"
{
    void FreeCudaMem(double *data);
    void GetCudaMemInfo(size_t *total_mem, size_t *free_mem);
}

void FreeCudaMem(double *data)
{
    cudaFree(data);
}

void GetCudaMemInfo(size_t *total_mem, size_t *free_mem)
{
    cudaMemGetInfo(free_mem, total_mem);
}