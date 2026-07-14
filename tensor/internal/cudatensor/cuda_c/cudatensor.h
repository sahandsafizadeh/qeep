#ifndef CUDATENSOR_H
#define CUDATENSOR_H

/*---------------- types ------------------*/
#include <stddef.h>
#include "types.h"

/*------------- initializers --------------*/
double *Full(double value, CUDAView view_o);
double *Eye(CUDAView view_o);
double *RandU(double l, double u, CUDAView view_o);
double *RandN(double u, double s, CUDAView view_o);
double *Of(double *input_data, CUDAView view_o);
double *From(CUDATensor t, CUDAView view_o);
double *Concat(CUDATensor ts[], size_t size, int dim, CUDAView view_o);

/*--------------- accessors ---------------*/
double At(CUDATensor t, DimArr index);
double *Patch(CUDATensor t, CUDATensor u, RangeArr ranges, CUDAView view_o);

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
double *Dot(CudaData a, CudaData b, DimArr dims_src, DimArr dims_dst);
double *MatMul(CudaData a, CudaData b, DimArr dims_a, DimArr dims_b, DimArr dims_c);

/*--------------- memory -----------------*/
void GetCudaMemInfo(size_t *free_mem, size_t *total_mem);
void FreeCudaMem(double *data);

#endif // CUDATENSOR_H