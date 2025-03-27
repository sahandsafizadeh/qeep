#include <math.h>
#include "cudatensor.h"

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

__device__ int getThreadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ int getGridStepSize()
{
    return gridDim.x * blockDim.x;
}

__global__ void applyUnaryFuncElemWise(
    double *x,
    double *y,
    size_t n,
    scalarUnaryFunc suf)
{
    const int tp = getThreadPosition();
    const int gss = getGridStepSize();

    for (size_t i = tp; i < n; i += gss)
    {
        y[i] = (*suf)(x[i]);
    }
}

__global__ void applyBinaryFuncElemWise(
    double *a,
    double *b,
    double *c,
    size_t n,
    scalarBinaryFunc sbf)
{
    const int tp = getThreadPosition();
    const int gss = getGridStepSize();

    for (size_t i = tp; i < n; i += gss)
    {
        c[i] = (*sbf)(a[i], b[i]);
    }
}

__global__ void applyHalfBinaryFuncElemWise(
    double *x,
    double a,
    double *y,
    size_t n,
    scalarBinaryFunc sbf)
{
    const int tp = getThreadPosition();
    const int gss = getGridStepSize();

    for (size_t i = tp; i < n; i += gss)
    {
        y[i] = (*sbf)(x[i], a);
    }
}

/* ----- API functions ----- */

double *Scale(double *x, double a, size_t n)
{
    double *y;
    cudaMalloc(&y, n * sizeof(double));
    applyHalfBinaryFuncElemWise<<<BLOCKS, THREADS>>>(x, a, y, n, scale_);
    return y;
}

double *Pow(double *x, double a, size_t n)
{
    double *y;
    cudaMalloc(&y, n * sizeof(double));
    applyHalfBinaryFuncElemWise<<<BLOCKS, THREADS>>>(x, a, y, n, pow_);
    return y;
}

double *Exp(double *x, size_t n)
{
    double *y;
    cudaMalloc(&y, n * sizeof(double));
    applyUnaryFuncElemWise<<<BLOCKS, THREADS>>>(x, y, n, exp_);
    return y;
}

double *Add(double *a, double *b, size_t n)
{
    double *c;
    cudaMalloc(&c, n * sizeof(double));
    applyBinaryFuncElemWise<<<BLOCKS, THREADS>>>(a, b, c, n, add_);
    return c;
}

double *Mul(double *a, double *b, size_t n)
{
    double *c;
    cudaMalloc(&c, n * sizeof(double));
    applyBinaryFuncElemWise<<<BLOCKS, THREADS>>>(a, b, c, n, mul_);
    return c;
}