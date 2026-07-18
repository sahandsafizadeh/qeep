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
double *Concat(CUDATensor ts[], int size, int dim, CUDAView view_o);

/*--------------- accessors ---------------*/
double At(CUDATensor t, DimArr index);
double *Patch(CUDATensor t, RangeArr ranges, CUDATensor u, CUDAView view_o);

/*--------------- reducers ----------------*/
double Sum(CUDATensor t);
double Max(CUDATensor t);
double Min(CUDATensor t);
double Avg(CUDATensor t);
double Var(CUDATensor t);
double Std(CUDATensor t);
double *Argmax(CUDATensor t, int dim, CUDAView view_o);
double *Argmin(CUDATensor t, int dim, CUDAView view_o);
double *SumAlong(CUDATensor t, int dim, CUDAView view_o);
double *MaxAlong(CUDATensor t, int dim, CUDAView view_o);
double *MinAlong(CUDATensor t, int dim, CUDAView view_o);
double *AvgAlong(CUDATensor t, int dim, CUDAView view_o);
double *VarAlong(CUDATensor t, int dim, CUDAView view_o);
double *StdAlong(CUDATensor t, int dim, CUDAView view_o);

/*--------------- operators ---------------*/
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

/*--------------- memory -----------------*/
void GetCudaMemInfo(size_t *free_mem, size_t *total_mem);
void FreeCudaMem(double *data);

#endif // CUDATENSOR_H