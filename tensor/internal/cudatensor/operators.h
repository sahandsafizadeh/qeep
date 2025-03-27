#ifndef OPERATORS_H
#define OPERATORS_H

#include <stddef.h>

double *Scale(double *x, double a, size_t n);
double *Pow(double *x, double a, size_t n);
double *Exp(double *x, size_t n);
double *Add(double *a, double *b, size_t n);
double *Mul(double *a, double *b, size_t n);

#endif // OPERATORS_H
