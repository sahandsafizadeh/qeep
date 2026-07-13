#ifndef TYPES
#define TYPES

#include <stddef.h>

#define MAX_DIMS 6

typedef struct DimArr
{
    int arr[MAX_DIMS];
    size_t size;
} DimArr;

typedef struct CUDAView
{
    int ofst;
    DimArr strd;
    DimArr dims;
} CUDAView;

typedef struct CUDAData
{
    double *arr;
    size_t size;
} CUDAData;

typedef struct CUDATensor
{
    CUDAView view;
    CUDAData data;
} CUDATensor;

/* ---------- range ---------- */

typedef struct Range
{
    int from;
    int to;
} Range;

typedef struct RangeArr
{
    Range arr[MAX_DIMS];
    size_t size;
} RangeArr;

#endif // TYPES