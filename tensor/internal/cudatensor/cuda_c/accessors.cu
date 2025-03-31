#include "common.h"

/* ----- API functions ----- */

extern "C"
{
    double At(const double *data, size_t index);
}

double At(const double *data, size_t index)
{
    double elem;
    handleCudaError(
        cudaMemcpy(
            &elem,
            &data[index],
            sizeof(double),
            cudaMemcpyDeviceToHost));

    return elem;
}