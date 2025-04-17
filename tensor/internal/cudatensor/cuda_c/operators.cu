#include <math.h>

#include "common.h"

/* ----- device functions ----- */

enum OperationType
{
    OP_SCALE,
    OP_POW,
    OP_EXP,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_SINH,
    OP_COSH,
    OP_TANH,
    OP_EQ,
    OP_NE,
    OP_GT,
    OP_GE,
    OP_LT,
    OP_LE,
    OP_ELMAX,
    OP_ELMIN,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
};

const double DOUBLE_EQUALITY_THRESHOLD = 1e-240;

__device__ double halfBinaryOp(double x, double a, OperationType opt)
{
    switch (opt)
    {
    case OP_SCALE:
        return a * x;
    case OP_POW:
        return pow(x, a);
    }

    return NAN;
}

__device__ double unaryOp(double x, OperationType opt)
{
    switch (opt)
    {
    case OP_EXP:
        return exp(x);
    case OP_LOG:
        return log(x);
    case OP_SIN:
        return sin(x);
    case OP_COS:
        return cos(x);
    case OP_TAN:
        return tan(x);
    case OP_SINH:
        return sinh(x);
    case OP_COSH:
        return cosh(x);
    case OP_TANH:
        return tanh(x);
    }

    return NAN;
}

__device__ double binaryOp(double a, double b, OperationType opt)
{
    switch (opt)
    {
    case OP_EQ:
        return abs(a - b) <= DOUBLE_EQUALITY_THRESHOLD ? 1. : 0.;
    case OP_NE:
        return abs(a - b) <= DOUBLE_EQUALITY_THRESHOLD ? 0. : 1.;
    case OP_GT:
        return a > b ? 1. : 0.;
    case OP_GE:
        return a >= b ? 1. : 0.;
    case OP_LT:
        return a < b ? 1. : 0.;
    case OP_LE:
        return a <= b ? 1. : 0.;
    case OP_ELMAX:
        return max(a, b);
    case OP_ELMIN:
        return min(a, b);
    case OP_ADD:
        return a + b;
    case OP_SUB:
        return a - b;
    case OP_MUL:
        return a * b;
    case OP_DIV:
        return a / b;
    }

    return NAN;
}

__device__ inline unsigned int threadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline unsigned int totalThreads()
{
    return gridDim.x * blockDim.x;
}

__global__ void applyHalfBinaryFuncElemWise(
    double *dst,
    const double *src1,
    size_t n,
    double srcc,
    OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        dst[i] = halfBinaryOp(src1[i], srcc, opt);
    }
}

__global__ void applyUnaryFuncElemWise(
    double *dst,
    const double *src,
    size_t n,
    OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        dst[i] = unaryOp(src[i], opt);
    }
}

__global__ void applyBinaryFuncElemWise(
    double *dst,
    const double *src1,
    const double *src2,
    size_t n,
    OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < n; i += stride)
    {
        dst[i] = binaryOp(src1[i], src2[i], opt);
    }
}

/* ----- API helper functions ----- */

double *runHalfBinaryOp(const double *x, size_t n, double a, OperationType opt)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyHalfBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(y, x, n, a, opt);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *runUnaryOp(const double *x, size_t n, OperationType opt)
{
    double *y;
    handleCudaError(
        cudaMalloc(&y, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyUnaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(y, x, n, opt);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y;
}

double *runBinaryOp(const double *a, const double *b, size_t n, OperationType opt)
{
    double *c;
    handleCudaError(
        cudaMalloc(&c, n * sizeof(double)));

    LaunchParams lps = launchParams(n);

    applyBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(c, a, b, n, opt);

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
    double *ElMax(const double *a, const double *b, size_t n);
    double *ElMin(const double *a, const double *b, size_t n);
    double *Add(const double *a, const double *b, size_t n);
    double *Sub(const double *a, const double *b, size_t n);
    double *Mul(const double *a, const double *b, size_t n);
    double *Div(const double *a, const double *b, size_t n);
}

double *Scale(const double *x, size_t n, double a)
{
    return runHalfBinaryOp(x, n, a, OP_SCALE);
}

double *Pow(const double *x, size_t n, double a)
{
    return runHalfBinaryOp(x, n, a, OP_POW);
}

double *Exp(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_EXP);
}

double *Log(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_LOG);
}

double *Sin(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_SIN);
}

double *Cos(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_COS);
}

double *Tan(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_TAN);
}

double *Sinh(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_SINH);
}

double *Cosh(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_COSH);
}

double *Tanh(const double *x, size_t n)
{
    return runUnaryOp(x, n, OP_TANH);
}

double *Eq(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_EQ);
}

double *Ne(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_NE);
}

double *Gt(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_GT);
}

double *Ge(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_GE);
}

double *Lt(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_LT);
}

double *Le(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_LE);
}

double *ElMax(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_ELMAX);
}

double *ElMin(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_ELMIN);
}

double *Add(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_ADD);
}

double *Sub(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_SUB);
}

double *Mul(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_MUL);
}

double *Div(const double *a, const double *b, size_t n)
{
    return runBinaryOp(a, b, n, OP_DIV);
}