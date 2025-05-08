#ifndef TYPES
#define TYPES

#include <stddef.h>

// two units higher than the actual limit for operations like UncontractedMatMul
#define MAX_DIMS 10

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