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

/*------------ shape modifiers ------------*/
double *Reshape(CudaData src);
double *Transpose(CudaData src, DimArr dims_src, DimArr dims_dst);
double *Broadcast(CudaData src, DimArr dims_src, DimArr dims_dst);

/*--------------- reducers ----------------*/
double Sum(CudaData src);
double Max(CudaData src);
double Min(CudaData src);
double Var(CudaData src);
double *Argmax(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *Argmin(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *SumAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *MaxAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *MinAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *AvgAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *VarAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);
double *StdAlong(CudaData src, int dim, DimArr dims_src, DimArr dims_dst);

/*--------------- operators ---------------*/
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

/*--------------- memory -----------------*/
void GetCudaMemInfo(size_t *free_mem, size_t *total_mem);
void FreeCudaMem(double *data);

#endif // CUDATENSOR_H