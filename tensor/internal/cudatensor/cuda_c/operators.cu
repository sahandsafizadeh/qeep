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

__global__ void applyHalfBinaryFuncElemWise(CUDATensor y, CUDATensor x, double a, OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < y.data.size; i += stride)
    {
        DimArr index = lnpos2index(i, y.view);
        int lnpos_x = index2lnpos(index, x.view);

        y.data.arr[i] = halfBinaryOp(x.data.arr[lnpos_x], a, opt);
    }
}

__global__ void applyUnaryFuncElemWise(CUDATensor y, CUDATensor x, OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < y.data.size; i += stride)
    {
        DimArr index = lnpos2index(i, y.view);
        int lnpos_x = index2lnpos(index, x.view);

        y.data.arr[i] = unaryOp(x.data.arr[lnpos_x], opt);
    }
}

__global__ void applyBinaryFuncElemWise(CUDATensor y, CUDATensor a, CUDATensor b, OperationType opt)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < y.data.size; i += stride)
    {
        DimArr index = lnpos2index(i, y.view);
        int lnpos_a = index2lnpos(index, a.view);
        int lnpos_b = index2lnpos(index, b.view);

        y.data.arr[i] = binaryOp(a.data.arr[lnpos_a], b.data.arr[lnpos_b], opt);
    }
}

__global__ void applyDot(CUDATensor y, CUDATensor a, CUDATensor b)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < y.data.size; i += stride)
    {
        DimArr index_y = lnpos2index(i, y.view);
        DimArr index_src = index_y;
        index_src.size = index_y.size + 1;

        size_t n = index_src.size;
        size_t cdim = a.view.dims.arr[n - 1];

        double temp = 0.;
        for (size_t j = 0; j < cdim; j++)
        {
            index_src.arr[n - 1] = j;
            int lnpos_a = index2lnpos(index_src, a.view);
            int lnpos_b = index2lnpos(index_src, b.view);

            temp += a.data.arr[lnpos_a] * b.data.arr[lnpos_b];
        }

        y.data.arr[i] = temp;
    }
}

__global__ void applyMatMul(CUDATensor y, CUDATensor a, CUDATensor b)
{
    const unsigned int tpos = threadPosition();
    const unsigned int stride = totalThreads();

    for (size_t i = tpos; i < y.data.size; i += stride)
    {
        DimArr index_a = lnpos2index(i, y.view);
        DimArr index_b = index_a;

        size_t n = index_a.size;
        size_t cdim = a.view.dims.arr[n - 1];

        double temp = 0.;
        for (size_t j = 0; j < cdim; j++)
        {
            index_a.arr[n - 1] = j;
            index_b.arr[n - 2] = j;
            int lnpos_a = index2lnpos(index_a, a.view);
            int lnpos_b = index2lnpos(index_b, b.view);

            temp += a.data.arr[lnpos_a] * b.data.arr[lnpos_b];
        }

        y.data.arr[i] = temp;
    }
}

/* ----- API helper functions ----- */

double *runHalfBinaryOp(CUDATensor x, double a, OperationType opt, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyHalfBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(o, x, a, opt);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *runUnaryOp(CUDATensor x, OperationType opt, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyUnaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(o, x, opt);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *runBinaryOp(CUDATensor a, CUDATensor b, OperationType opt, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyBinaryFuncElemWise<<<lps.blockSize, lps.threadSize>>>(o, a, b, opt);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *runDot(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyDot<<<lps.blockSize, lps.threadSize>>>(o, a, b);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

double *runMatMul(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    size_t n = elemcnt(view_o.dims);

    CUDAData data_o = (CUDAData){NULL, n};
    handleCudaError(
        cudaMalloc(&data_o.arr, data_o.size * sizeof(double)));

    CUDATensor o = (CUDATensor){view_o, data_o};

    LaunchParams lps = launchParams(o.data.size);
    applyMatMul<<<lps.blockSize, lps.threadSize>>>(o, a, b);
    handleCudaError(
        cudaGetLastError());

    return o.data.arr;
}

/* ----- API functions ----- */

extern "C"
{
    double *Scale(CUDATensor x, double a, CUDAView view_o);
    double *Pow(CUDATensor x, double a, CUDAView view_o);
    double *Exp(CUDATensor x, CUDAView view_o);
    double *Log(CUDATensor x, CUDAView view_o);
    double *Sin(CUDATensor x, CUDAView view_o);
    double *Cos(CUDATensor x, CUDAView view_o);
    double *Tan(CUDATensor x, CUDAView view_o);
    double *Sinh(CUDATensor x, CUDAView view_o);
    double *Cosh(CUDATensor x, CUDAView view_o);
    double *Tanh(CUDATensor x, CUDAView view_o);
    double *Eq(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Ne(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Gt(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Ge(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Lt(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Le(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *ElMax(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *ElMin(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Add(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Sub(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Mul(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Div(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *Dot(CUDATensor a, CUDATensor b, CUDAView view_o);
    double *MatMul(CUDATensor a, CUDATensor b, CUDAView view_o);
}

double *Scale(CUDATensor x, double a, CUDAView view_o)
{
    return runHalfBinaryOp(x, a, OP_SCALE, view_o);
}

double *Pow(CUDATensor x, double a, CUDAView view_o)
{
    return runHalfBinaryOp(x, a, OP_POW, view_o);
}

double *Exp(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_EXP, view_o);
}

double *Log(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_LOG, view_o);
}

double *Sin(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_SIN, view_o);
}

double *Cos(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_COS, view_o);
}

double *Tan(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_TAN, view_o);
}

double *Sinh(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_SINH, view_o);
}

double *Cosh(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_COSH, view_o);
}

double *Tanh(CUDATensor x, CUDAView view_o)
{
    return runUnaryOp(x, OP_TANH, view_o);
}

double *Eq(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_EQ, view_o);
}

double *Ne(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_NE, view_o);
}

double *Gt(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_GT, view_o);
}

double *Ge(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_GE, view_o);
}

double *Lt(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_LT, view_o);
}

double *Le(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_LE, view_o);
}

double *ElMax(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_ELMAX, view_o);
}

double *ElMin(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_ELMIN, view_o);
}

double *Add(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_ADD, view_o);
}

double *Sub(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_SUB, view_o);
}

double *Mul(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_MUL, view_o);
}

double *Div(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runBinaryOp(a, b, OP_DIV, view_o);
}

double *Dot(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runDot(a, b, view_o);
}

double *MatMul(CUDATensor a, CUDATensor b, CUDAView view_o)
{
    return runMatMul(a, b, view_o);
}