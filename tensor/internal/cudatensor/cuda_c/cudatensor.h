#ifndef CUDATENSOR_H
#define CUDATENSOR_H

#include <stddef.h>

double *Full(size_t n, double value);
double *Eye(size_t n, size_t d);
double *RandU(size_t n, double l, double u);
double *RandN(size_t n, double u, double s);
double *Of(size_t n, const double input_data[]);

void FreeCudaMem(double *data);

#endif // CUDATENSOR_H