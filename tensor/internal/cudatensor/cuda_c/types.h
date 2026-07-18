#ifndef TYPES
#define TYPES

#include <stddef.h>

#define MAX_DIMS 6

typedef struct DimArr
{
    int size;
    size_t arr[MAX_DIMS];
} DimArr;

typedef struct CUDAView
{
    size_t ofst;
    DimArr strd;
    DimArr dims;
} CUDAView;

typedef struct CUDAData
{
    size_t size;
    double *arr;
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
    int size;
    Range arr[MAX_DIMS];
} RangeArr;

#endif // TYPES