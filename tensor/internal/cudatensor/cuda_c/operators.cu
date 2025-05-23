#include <math.h>

#include "types.h"
#include "common.cuh"
#include "devcommon.cuh"

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

/* ----- device functions ----- */

__device__ inline double halfBinaryOp(double x, double a, OperationType opt)
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

__device__ inline double unaryOp(double x, OperationType opt)
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

__device__ inline double binaryOp(double a, double b, OperationType opt)
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

__global__ void applyHalfBinaryFuncElemWise(CudaData dst, CudaData src1, double srcc, OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        dst.arr[i] = halfBinaryOp(src1.arr[i], srcc, opt);
    }
}

__global__ void applyUnaryFuncElemWise(CudaData dst, CudaData src, OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        dst.arr[i] = unaryOp(src.arr[i], opt);
    }
}

__global__ void applyBinaryFuncElemWise(CudaData dst, CudaData src1, CudaData src2, OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        dst.arr[i] = binaryOp(src1.arr[i], src2.arr[i], opt);
    }
}

__global__ void applyDot(
    CudaData dst,
    CudaData src1,
    CudaData src2,
    DimArr rcp_dst,
    DimArr rcp_src,
    int cdim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        size_t n;
        int lnpos_dst;
        int lnpos_src;
        DimArr index_dst;
        DimArr index_src;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src = index_dst;

        n = index_dst.size + 1;
        index_src.size = n;

        double temp = 0.;
        for (size_t j = 0; j < cdim; j++)
        {
            index_src.arr[n - 1] = j;
            lnpos_src = encode(index_src, rcp_src);

            temp += src1.arr[lnpos_src] * src2.arr[lnpos_src];
        }

        dst.arr[lnpos_dst] = temp;
    }
}

__global__ void applyMatMul(
    CudaData dst,
    CudaData src1,
    CudaData src2,
    DimArr rcp_dst,
    DimArr rcp_src1,
    DimArr rcp_src2,
    int cdim)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < dst.size; i += stride)
    {
        size_t n;
        int lnpos_dst;
        int lnpos_src1;
        int lnpos_src2;
        DimArr index_dst;
        DimArr index_src1;
        DimArr index_src2;

        lnpos_dst = i;
        index_dst = decode(lnpos_dst, rcp_dst);
        index_src1 = index_dst;
        index_src2 = index_dst;

        n = index_dst.size;
        index_src1.arr[n - 2] = index_dst.arr[n - 2];
        index_src2.arr[n - 1] = index_dst.arr[n - 1];

        double temp = 0.;
        for (size_t j = 0; j < cdim; j++)
        {
            index_src1.arr[n - 1] = j;
            index_src2.arr[n - 2] = j;
            lnpos_src1 = encode(index_src1, rcp_src1);
            lnpos_src2 = encode(index_src2, rcp_src2);

            temp += src1.arr[lnpos_src1] * src2.arr[lnpos_src2];
        }

        dst.arr[lnpos_dst] = temp;
    }
}

/* ----- API helper functions ----- */

double *runHalfBinaryOp(CudaData x, double a, OperationType opt)
{
    CudaData y = (CudaData){NULL, x.size};
    handleCudaError(
        cudaMalloc(&y.arr, y.size * sizeof(double)));

    LaunchParams lps = launchParams(y.size);

    applyHalfBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(y, x, a, opt);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y.arr;
}

double *runUnaryOp(CudaData x, OperationType opt)
{
    CudaData y = (CudaData){NULL, x.size};
    handleCudaError(
        cudaMalloc(&y.arr, y.size * sizeof(double)));

    LaunchParams lps = launchParams(y.size);

    applyUnaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(y, x, opt);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return y.arr;
}

double *runBinaryOp(CudaData a, CudaData b, OperationType opt)
{
    CudaData c = (CudaData){NULL, a.size};
    handleCudaError(
        cudaMalloc(&c.arr, c.size * sizeof(double)));

    LaunchParams lps = launchParams(c.size);

    applyBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(c, a, b, opt);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return c.arr;
}

double *runDot(CudaData a, CudaData b, DimArr dims_src, DimArr dims_dst)
{
    size_t n = elemcnt(dims_dst);
    DimArr rcp_dst = rcumprod(dims_dst);
    DimArr rcp_src = rcumprod(dims_src);
    size_t cdim = dims_src.arr[dims_src.size - 1];

    CudaData c = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&c.arr, c.size * sizeof(double)));

    LaunchParams lps = launchParams(c.size);

    applyDot<<<lps.blockSize, lps.threadSize>>>(c, a, b, rcp_dst, rcp_src, cdim);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return c.arr;
}

double *runMatMul(CudaData a, CudaData b, DimArr dims_a, DimArr dims_b, DimArr dims_c)
{
    size_t n = elemcnt(dims_c);
    DimArr rcp_c = rcumprod(dims_c);
    DimArr rcp_a = rcumprod(dims_a);
    DimArr rcp_b = rcumprod(dims_b);
    size_t cdim = dims_a.arr[dims_a.size - 1];

    CudaData c = (CudaData){NULL, n};
    handleCudaError(
        cudaMalloc(&c.arr, c.size * sizeof(double)));

    LaunchParams lps = launchParams(c.size);

    applyMatMul<<<lps.blockSize, lps.threadSize>>>(c, a, b, rcp_c, rcp_a, rcp_b, cdim);

    handleCudaError(
        cudaGetLastError());
    handleCudaError(
        cudaDeviceSynchronize());

    return c.arr;
}

/* ----- API functions ----- */

extern "C"
{
    double *Scale(CudaData x, double a);
    double *Pow(CudaData x, double a);
    double *Exp(CudaData x);
    double *Log(CudaData x);
    double *Sin(CudaData x);
    double *Cos(CudaData x);
    double *Tan(CudaData x);
    double *Sinh(CudaData x);
    double *Cosh(CudaData x);
    double *Tanh(CudaData x);
    double *Eq(CudaData a, CudaData b);
    double *Ne(CudaData a, CudaData b);
    double *Gt(CudaData a, CudaData b);
    double *Ge(CudaData a, CudaData b);
    double *Lt(CudaData a, CudaData b);
    double *Le(CudaData a, CudaData b);
    double *ElMax(CudaData a, CudaData b);
    double *ElMin(CudaData a, CudaData b);
    double *Add(CudaData a, CudaData b);
    double *Sub(CudaData a, CudaData b);
    double *Mul(CudaData a, CudaData b);
    double *Div(CudaData a, CudaData b);
    double *Dot(CudaData a, CudaData b, DimArr dims_src, DimArr dims_dst);
    double *MatMul(CudaData a, CudaData b, DimArr dims_a, DimArr dims_b, DimArr dims_c);
}

double *Scale(CudaData x, double a)
{
    return runHalfBinaryOp(x, a, OP_SCALE);
}

double *Pow(CudaData x, double a)
{
    return runHalfBinaryOp(x, a, OP_POW);
}

double *Exp(CudaData x)
{
    return runUnaryOp(x, OP_EXP);
}

double *Log(CudaData x)
{
    return runUnaryOp(x, OP_LOG);
}

double *Sin(CudaData x)
{
    return runUnaryOp(x, OP_SIN);
}

double *Cos(CudaData x)
{
    return runUnaryOp(x, OP_COS);
}

double *Tan(CudaData x)
{
    return runUnaryOp(x, OP_TAN);
}

double *Sinh(CudaData x)
{
    return runUnaryOp(x, OP_SINH);
}

double *Cosh(CudaData x)
{
    return runUnaryOp(x, OP_COSH);
}

double *Tanh(CudaData x)
{
    return runUnaryOp(x, OP_TANH);
}

double *Eq(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_EQ);
}

double *Ne(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_NE);
}

double *Gt(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_GT);
}

double *Ge(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_GE);
}

double *Lt(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_LT);
}

double *Le(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_LE);
}

double *ElMax(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_ELMAX);
}

double *ElMin(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_ELMIN);
}

double *Add(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_ADD);
}

double *Sub(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_SUB);
}

double *Mul(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_MUL);
}

double *Div(CudaData a, CudaData b)
{
    return runBinaryOp(a, b, OP_DIV);
}

double *Dot(CudaData a, CudaData b, DimArr dims_src, DimArr dims_dst)
{
    return runDot(a, b, dims_src, dims_dst);
}

double *MatMul(CudaData a, CudaData b, DimArr dims_a, DimArr dims_b, DimArr dims_c)
{
    return runMatMul(a, b, dims_a, dims_b, dims_c);
}