#include "CudaOps.cuh"

//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void squareKernel(int* a, int* b)
{
    int i = threadIdx.x;
    b[i] = a[i] *a[i];
}

__global__ void step(Ray* rayptr, Volume *volume) {
    int index = blockIdx.x * RAYS_PER_DIM + threadIdx.x;  //This fucks shit up if RPD > 1024!!

    if (!rayptr[index].full) {
        float x_ = rayptr[index].origin[0] * rayptr[index].step_vector[0] * 1;
        float y_ = rayptr[index].origin[1] * rayptr[index].step_vector[1] * 1;
        float z_ = rayptr[index].origin[2] * rayptr[index].step_vector[2] * 1;
        rayptr[index].acc_color = volume[0].blocks[(int)x_][(int)y_][(int)z_].color;
        rayptr[index].acc_alpha = volume[0].blocks[(int)x_][(int)y_][(int)z_].alpha;
    }
}


CudaOperator::CudaOperator(){
    cout << NUM_RAYS * sizeof(Ray) << endl << sizeof(Volume) << endl;
    cudaMallocManaged(&rayptr, NUM_RAYS * sizeof(Ray));
    cudaMallocManaged(&volume, sizeof(Volume));
    cout << "Cuda initialized" << endl;
    }


void CudaOperator::update(Ray *rp) {
    rayptr = rp;
}

void CudaOperator::doStuff() {
    int* a, * b;
    int SIZE = 20;
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
