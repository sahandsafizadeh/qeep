#include "common.h"

extern "C"
{
    void FreeCudaMem(double *data);
}

void FreeCudaMem(double *data)
{
    cudaFree(data);
}