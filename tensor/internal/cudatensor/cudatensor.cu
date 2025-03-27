void FreeCUDAMem(double *data)
{
    cudaFree(data);
}