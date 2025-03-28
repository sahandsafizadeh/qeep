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

__global__ void applyUnaryFuncElemWise(
    double *x,
    double *y,
    size_t n,
    scalarUnaryFunc suf)
{
    const int tpos = getThreadPosition();
    const int gstep = getGridStepSize();

    for (size_t i = tpos; i < n; i += gstep)
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
    const int tpos = getThreadPosition();
    const int gstep = getGridStepSize();

    for (size_t i = tpos; i < n; i += gstep)
    {
        c[i] = (*sbf)(a[i], b[i]);
    }
}

__global__ void applyHalfBinaryFuncElemWise(
    double *x,
    double *y,
    double a,
    size_t n,
    scalarBinaryFunc sbf)
{
    const int tpos = getThreadPosition();
    const int gstep = getGridStepSize();

    for (size_t i = tpos; i < n; i += gstep)
    {
        y[i] = (*sbf)(x[i], a);
    }
}

/* ----- API functions ----- */

extern "C"
{
    double *Scale(double *x, double a, size_t n);
    double *Pow(double *x, double a, size_t n);
    double *Exp(double *x, size_t n);
    double *Add(double *a, double *b, size_t n);
    double *Mul(double *a, double *b, size_t n);
}

double *Scale(double *x, double a, size_t n)
{
    double *y;
    cudaMalloc(&y, n * sizeof(double));

    applyHalfBinaryFuncElemWise<<<BLOCKS, THREADS>>>(x, y, a, n, scale_);
    cudaDeviceSynchronize();

    return y;
}

double *Pow(double *x, double a, size_t n)
{
    double *y;
    cudaMalloc(&y, n * sizeof(double));

    applyHalfBinaryFuncElemWise<<<BLOCKS, THREADS>>>(x, y, a, n, pow_);
    cudaDeviceSynchronize();

    return y;
}

double *Exp(double *x, size_t n)
{
    double *y;
    cudaMalloc(&y, n * sizeof(double));

    applyUnaryFuncElemWise<<<BLOCKS, THREADS>>>(x, y, n, exp_);
    cudaDeviceSynchronize();

    return y;
}

double *Add(double *a, double *b, size_t n)
{
    double *c;
    cudaMalloc(&c, n * sizeof(double));

    applyBinaryFuncElemWise<<<BLOCKS, THREADS>>>(a, b, c, n, add_);
    cudaDeviceSynchronize();

    return c;
}

double *Mul(double *a, double *b, size_t n)
{
    double *c;
    cudaMalloc(&c, n * sizeof(double));

    applyBinaryFuncElemWise<<<BLOCKS, THREADS>>>(a, b, c, n, mul_);
    cudaDeviceSynchronize();

    return c;
}