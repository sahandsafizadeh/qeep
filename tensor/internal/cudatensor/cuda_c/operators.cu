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

__device__ inline unsigned int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline unsigned int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__global__ void applyUnaryFuncElemWise(
    double *dst,
    const double *src,
    size_t n,
    scalarUnaryFunc suf)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        dst[i] = suf(src[i]);
    }
}

__global__ void applyBinaryFuncElemWise(
    double *dst,
    const double *src1,
    const double *src2,
    size_t n,
    scalarBinaryFunc sbf)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        dst[i] = sbf(src1[i], src2[i]);
    }
}

__global__ void applyHalfBinaryFuncElemWise(
    double *dst,
    const double *src1,
    size_t n,
    double srcc,
    scalarBinaryFunc sbf)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        dst[i] = sbf(src1[i], srcc);
    }
}

/* ----- API helper functions ----- */

double *runHalfBinaryOp(const double *x, size_t n, double a, scalarBinaryFunc sbf)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyHalfBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(y, x, n, a, sbf);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *runUnaryOp(const double *x, size_t n, scalarUnaryFunc suf)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyUnaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(y, x, n, suf);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *runBinaryOp(const double *a, const double *b, size_t n, scalarBinaryFunc sbf)
{
    double *c;
    handleCudaError(
        cudaMalloc(&c, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(c, a, b, n, sbf);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return c;
}

/* ----- API functions ----- */

extern "C"
{
    double *Scale(const double *x, size_t n, double a);
    double *Pow(const double *x, size_t n, double a);
    double *Exp(const double *x, size_t n);
    double *Log(const double *x, size_t n);
    double *Sin(const double *x, size_t n);
    double *Cos(const double *x, size_t n);
    double *Tan(const double *x, size_t n);
    double *Sinh(const double *x, size_t n);
    double *Cosh(const double *x, size_t n);
    double *Tanh(const double *x, size_t n);
    double *Eq(const double *a, const double *b, size_t n);
    double *Ne(const double *a, const double *b, size_t n);
    double *Gt(const double *a, const double *b, size_t n);
    double *Ge(const double *a, const double *b, size_t n);
    double *Lt(const double *a, const double *b, size_t n);
    double *Le(const double *a, const double *b, size_t n);
    double *ElMin(const double *a, const double *b, size_t n);
    double *ElMax(const double *a, const double *b, size_t n);
    double *Add(const double *a, const double *b, size_t n);
    double *Sub(const double *a, const double *b, size_t n);
    double *Mul(const double *a, const double *b, size_t n);
    double *Div(const double *a, const double *b, size_t n);
}

double *Scale(const double *x, size_t n, double a)
{
    return runHalfBinaryOp(x, n, a, scale_);
}

double *Pow(const double *x, size_t n, double a)
{
    return runHalfBinaryOp(x, n, a, pow_);
}

double *Exp(const double *x, size_t n)
{
    return runUnaryOp(x, n, exp_);
}

double *Log(const double *x, size_t n)
{
    return runUnaryOp(x, n, log_);
}

double *Sin(const double *x, size_t n)
{
    return runUnaryOp(x, n, sin_);
}

double *Cos(const double *x, size_t n)
{
    return runUnaryOp(x, n, cos_);
}

double *Tan(const double *x, size_t n)
{
    return runUnaryOp(x, n, tan_);
}

double *Sinh(const double *x, size_t n)
{
    return runUnaryOp(x, n, sinh_);
}

double *Cosh(const double *x, size_t n)
{
    return runUnaryOp(x, n, cosh_);
}

double *Tanh(const double *x, size_t n)
{
    return runUnaryOp(x, n, tanh_);
}

double *Eq(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, eq_);
}

double *Ne(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, ne_);
}

double *Gt(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, gt_);
}

double *Ge(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, ge_);
}

double *Lt(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, lt_);
}

double *Le(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, le_);
}

double *ElMin(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, elmin_);
}

double *ElMax(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, elmax_);
}

double *Add(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, add_);
}

double *Sub(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, sub_);
}

double *Mul(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, mul_);
}

double *Div(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, div_);
}