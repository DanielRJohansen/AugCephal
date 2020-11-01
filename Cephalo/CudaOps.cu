#include "CudaOps.cuh"

#define vol_x_range VOL_X
#define vol_y_range VOL_Y
#define vol_z_range VOL_Z
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void squareKernel(int* a, int* b)
{
    int i = threadIdx.x;
    b[i] = a[i] *a[i];
}

__global__ void stepKernel(Ray* rayptr, Volume *volume) {
    int index = blockIdx.x * RAYS_PER_DIM + threadIdx.x;  //This fucks shit up if RPD > 1024!!

    for (int step = 0; step < 1000; step++) {
        if (!rayptr[index].full) {
            //Float3 step_pos = rayptr[index].origin + rayptr[index].step_vector * step;
            float x_ = rayptr[index].origin.x + rayptr[index].step_vector.x * 1;
            float y_ = rayptr[index].origin.y + rayptr[index].step_vector.y * 1;
            float z_ = rayptr[index].origin.z + rayptr[index].step_vector.z * 1;
            int vol_x = (int)x_ + vol_x_range / 2;
            int vol_y = (int)y_ + vol_y_range / 2;
            int vol_z = (int)z_ + vol_z_range / 2;
            if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && // Only proceed if coordinate is within volume!
                vol_x < vol_x_range && vol_y < vol_y_range && vol_z < vol_z_range) {
                int volume_index = vol_z * 256 * 256 + vol_y * 256 + vol_x;
                rayptr[index].acc_color += volume[0].blocks[volume_index].color;
                rayptr[index].acc_alpha += volume[0].blocks[volume_index].alpha;
                if (rayptr[index].acc_alpha >= 1) 
                    rayptr[index].full = true;
            }
        }
    }
}


CudaOperator::CudaOperator(){
    cout << (NUM_RAYS * sizeof(Ray))/1000000 << " MB" << endl << sizeof(Volume)/1000000 << " MB" <<endl;
    cudaMallocManaged(&rayptr, NUM_RAYS * sizeof(Ray));
    cudaMallocManaged(&volume, sizeof(Volume));
    cout << "Cuda initialized" << endl;
    }


void CudaOperator::update(Ray *rp) {
    rayptr = rp;
}

void CudaOperator::rayStep() {
    stepKernel << <RAYS_PER_DIM, RAYS_PER_DIM >> > (rayptr, volume);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();
    cout << rayptr[120].acc_color << "  " << rayptr[140].acc_color;
    cout << "Raystep complete "<< endl;
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
