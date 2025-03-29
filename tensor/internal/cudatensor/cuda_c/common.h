#ifndef COMMON_H
#define COMMON_H

typedef double(scalarUnaryFunc)(double);
typedef double(scalarBinaryFunc)(double, double);

int blockSize(size_t n);
int threadSize(size_t n);
void handleCudaError(cudaError_t err);

#endif // COMMON_H