#ifndef CUDATENSOR_H
#define CUDATENSOR_H

/*---------------- types ------------------*/
#include <stddef.h>
#include "types.h"

/*------------- initializers --------------*/
double *Full(size_t n, double value);
double *Eye(size_t n, size_t d);
double *RandU(size_t n, double l, double u);
double *RandN(size_t n, double u, double s);
double *Of(size_t n, double *input_data);

/*--------------- accessors ---------------*/
double At(CudaData src, DimArr dims, DimArr index);
double *Slice(CudaData src, DimArr dims, RangeArr index);
double *Patch(CudaData bas, DimArr dims, CudaData src, RangeArr index);

/*--------------- reducers ----------------*/
double Sum(const double *src, size_t n);
double Max(const double *src, size_t n);
double Min(const double *src, size_t n);

/*--------------- operators ---------------*/
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

/*--------------- memory -----------------*/
void GetCudaMemInfo(size_t *free_mem, size_t *total_mem);
void FreeCudaMem(double *data);

#endif // CUDATENSOR_H