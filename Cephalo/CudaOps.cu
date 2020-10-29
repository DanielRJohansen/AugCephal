#include "CudaOps.cuh"

//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void squareKernel(int* a, int* b)
{
    int i = threadIdx.x;
    b[i] = a[i] *a[i];
}


int bmain() {
    int* a, * b;
    int SIZE = 512;
    cudaMallocManaged(&a, SIZE * sizeof(int));
    cudaMallocManaged(&b, SIZE * sizeof(int));
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
    }
    b[0] = 0;
    squareKernel <<<1, 512 >> > (a, b);
    cudaDeviceSynchronize();
    for (int i = 0; i < SIZE; i++) {
        cout << b[i] << " ";
    }
    cudaFree(a);
    cudaFree(b);
    return 0;
}





void CudaOperator::doStuff() {
    int* a, * b;
    int SIZE = 512;
    cudaMallocManaged(&a, SIZE * sizeof(int));
    cudaMallocManaged(&b, SIZE * sizeof(int));
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
    }
    b[0] = 0;
    squareKernel << <1, 512 >> > (a, b);
    cudaDeviceSynchronize();
    for (int i = 0; i < SIZE; i++) {
        cout << b[i] << " ";
    }
    cudaFree(a);
    cudaFree(b);
}
