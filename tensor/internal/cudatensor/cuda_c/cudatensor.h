#ifndef CUDATENSOR_H
#define CUDATENSOR_H

#include <stddef.h>

double *Full(size_t n, double value);
void FreeCUDAMemory(double *dev_data);

double *Scale(double *x, double a, size_t n);
double *Pow(double *x, double a, size_t n);
double *Exp(double *x, size_t n);
double *Add(double *a, double *b, size_t n);
double *Mul(double *a, double *b, size_t n);

#endif // CUDATENSOR_H
