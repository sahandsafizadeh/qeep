#ifndef COMMON_H
#define COMMON_H

#define BLOCKS 256
#define THREADS 256

typedef double (*scalarUnaryFunc)(double);
typedef double (*scalarBinaryFunc)(double, double);

__device__ int getThreadPosition();
__device__ int getGridStepSize();

#endif // COMMON_H
