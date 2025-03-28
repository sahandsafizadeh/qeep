__device__ int getThreadPosition()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ int getGridStepSize()
{
    return gridDim.x * blockDim.x;
}
