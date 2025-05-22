#ifndef TYPES
#define TYPES

#include <stddef.h>

#define MAX_DIMS 6

typedef struct Range
{
    int from;
    int to;
} Range;

typedef struct CudaData
{
    double *arr;
    size_t size;
} CudaData;

typedef struct DimArr
{
    int arr[MAX_DIMS];
    size_t size;
} DimArr;

typedef struct RangeArr
{
    Range arr[MAX_DIMS];
    size_t size;
} RangeArr;

#endif // TYPES