#include "common.h"

/* ----- indexing functions ----- */

void rcumprod(int *rcp, const int *dims, size_t n)
{
    int prod = 1;
    for (size_t i = 0; i < n; i++)
    {
        size_t j = n - i - 1;
        rcp[j] = prod;
        prod *= dims[j];
    }
}

void rcumprod(int *rcp, const Range *range, size_t n)
{
    int prod = 1;
    for (size_t i = 0; i < n; i++)
    {
        size_t j = n - i - 1;
        rcp[j] = prod;
        prod *= range[j].to - range[j].from;
    }
}

void encode(int *lnpos, const int *index, const int *rcp, size_t n)
{
    *lnpos = 0;
    for (size_t i = 0; i < n; i++)
    {
        size_t j = n - i - 1;
        *lnpos += index[j] * rcp[j];
    }
}

/* ----- helper functions ----- */

int linearPosition(const int *dims, const int *index, size_t n)
{
    int lnpos = 0;
    int *rcp = (int *)(malloc(n * sizeof(int)));

    rcumprod(rcp, dims, n);
    encode(&lnpos, index, rcp, n);
    free(rcp);

    return lnpos;
}

/* ----- API functions ----- */

extern "C"
{
    double At(const double *data, const int *dims, const int *index, size_t n);
}

double At(const double *data, const int *dims, const int *index, size_t n)
{
    int lnpos = linearPosition(dims, index, n);

    double elem;
    handleCudaError(
        cudaMemcpy(
            &elem,
            &data[lnpos],
            sizeof(double),
            cudaMemcpyDeviceToHost));

    return elem;
}