#ifndef CUDATENSOR_H
#define CUDATENSOR_H

#include <stddef.h>

double *Full(size_t n, double value);
double *Eye(size_t n, size_t d);
double *RandU(size_t n, double l, double u);
double *RandN(size_t n, double u, double s);
double *Of(size_t n, const double *input_data);

double At(const double *data, size_t index);

double *Scale(const double *x, size_t n, double a);
double *Pow(const double *x, size_t n, double a);
double *Exp(const double *x, size_t n);
double *Add(const double *a, const double *b, size_t n);
double *Mul(const double *a, const double *b, size_t n);

void FreeCudaMem(double *data);
void GetCudaMemInfo(size_t *total_mem, size_t *free_mem);

#endif // CUDATENSOR_H