#include <math.h>

#include "common.h"

/* ----- scalar functions ----- */

const double DOUBLE_EQUALITY_THRESHOLD = 1e-240;

double scale_(double x, double a)
{
    return a * x;
}

double pow_(double x, double a)
{
    return pow(x, a);
}

double exp_(double x)
{
    return exp(x);
}

double log_(double x)
{
    return log(x);
}

double sin_(double x)
{
    return sin(x);
}

double cos_(double x)
{
    return cos(x);
}

double tan_(double x)
{
    return tan(x);
}

double sinh_(double x)
{
    return sinh(x);
}

double cosh_(double x)
{
    return cosh(x);
}

double tanh_(double x)
{
    return tanh(x);
}

double eq_(double a, double b)
{
    return abs(a - b) <= DOUBLE_EQUALITY_THRESHOLD ? 1. : 0.;
}

double ne_(double a, double b)
{
    return abs(a - b) <= DOUBLE_EQUALITY_THRESHOLD ? 0. : 1.;
}

double gt_(double a, double b)
{
    return a > b ? 1. : 0.;
}

double ge_(double a, double b)
{
    return a >= b ? 1. : 0.;
}

double lt_(double a, double b)
{
    return a < b ? 1. : 0.;
}

double le_(double a, double b)
{
    return a <= b ? 1. : 0.;
}

double elmax_(double a, double b)
{
    return max(a, b);
}

double elmin_(double a, double b)
{
    return min(a, b);
}

double add_(double a, double b)
{
    return a + b;
}

double sub_(double a, double b)
{
    return a - b;
}

double mul_(double a, double b)
{
    return a * b;
}

double div_(double a, double b)
{
    return a / b;
}

/* ----- device functions ----- */

typedef double(scalarUnaryFunc)(double);
typedef double(scalarBinaryFunc)(double, double);

__device__ inline int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__global__ void applyUnaryFuncElemWise(
    scalarUnaryFunc suf,
    const double *x,
    size_t n,
    double *y)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        y[i] = suf(x[i]);
    }
}

__global__ void applyBinaryFuncElemWise(
    scalarBinaryFunc sbf,
    const double *a,
    const double *b,
    size_t n,
    double *c)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        c[i] = sbf(a[i], b[i]);
    }
}

__global__ void applyHalfBinaryFuncElemWise(
    scalarBinaryFunc sbf,
    const double *x,
    size_t n,
    double a,
    double *y)
{
    const int tpos = threadPosition();
    const int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        y[i] = sbf(x[i], a);
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Scale(const double *x, size_t n, double a);
    double *Pow(const double *x, size_t n, double a);
    double *Exp(const double *x, size_t n);
    double *Add(const double *a, const double *b, size_t n);
    double *Mul(const double *a, const double *b, size_t n);
}

double *Scale(const double *x, size_t n, double a)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyHalfBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(scale_, x, n, a, y);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *Pow(const double *x, size_t n, double a)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyHalfBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(pow_, x, n, a, y);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *Exp(const double *x, size_t n)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyUnaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(exp_, x, n, y);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *Add(const double *a, const double *b, size_t n)
{
    double *c;
    handleCudaError(
        cudaMalloc(&c, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(add_, a, b, n, c);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return c;
}

double *Mul(const double *a, const double *b, size_t n)
{
    double *c;
    handleCudaError(
        cudaMalloc(&c, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(mul_, a, b, n, c);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return c;
}