#ifndef CUDATENSOR_H
#define CUDATENSOR_H

#include <stddef.h>
#include "types.h"

double *Full(size_t n, double value);
double *Eye(size_t n, size_t d);
double *RandU(size_t n, double l, double u);
double *RandN(size_t n, double u, double s);
double *Of(const double *input_data, size_t n);

double At(const double *data, const int *dims, const int *index, size_t n);
double *Slice(const double *src, const int *dims, const Range *index, size_t n);
double *Patch(const double *bas, const int *dims, const double *src, const Range *index, size_t n);

double *Scale(const double *x, size_t n, double a);
double *Pow(const double *x, size_t n, double a);
double *Exp(const double *x, size_t n);
double *Add(const double *a, const double *b, size_t n);
double *Mul(const double *a, const double *b, size_t n);

void GetCudaMemInfo(size_t *free_mem, size_t *total_mem);
void FreeCudaMem(double *data);

#endif // CUDATENSOR_H