#ifndef TYPES
#define TYPES

#include <stddef.h>

#define MAX_DIMS 6

typedef struct DimArr
{
    size_t arr[MAX_DIMS];
    size_t size;
} DimArr;

typedef struct CUDAView
{
    size_t ofst;
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
    size_t from;
    size_t to;
} Range;

typedef struct RangeArr
{
    Range arr[MAX_DIMS];
    size_t size;
} RangeArr;

#endif // TYPES