#include "CudaOps.cuh"

#define vol_x_range VOL_X
#define vol_y_range VOL_Y
#define vol_z_range VOL_Z
#define RAY_SS 1
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void squareKernel(int* a)
{
    int i = blockIdx.x * 512 + threadIdx.x;
    a[i] = i;
}

__global__ void stepKernel(Ray* rayptr, Block *blocks) {
    int index = blockIdx.x * RAYS_PER_DIM + threadIdx.x;  //This fucks shit up if RPD > 1024!!

    //Reset ray
    Ray ray = rayptr[index];
    rayptr[index].acc_color = 0;
    rayptr[index].full = false;

    float sin_pitch = sin(ray.cam_pitch);
    float cos_pitch = cos(ray.cam_pitch);
    float sin_yaw = sin(ray.cam_yaw);
    float cos_yaw = cos(ray.cam_yaw);

    float x = ray.rel_unit_vector.x;
    float y = ray.rel_unit_vector.y;
    float z = ray.rel_unit_vector.z;

    // Rotate rel vector around y
    float x_y = cos_pitch * x + sin_pitch * z;
    float z_y = -sin_pitch * x + cos_pitch * z;

    // Rotate relative vector about z
    float x_z = cos_yaw * x_y - sin_yaw * y;
    float y_z = sin_yaw * x_y + cos_yaw * y;
    
    float x_ = x_z * RAY_SS;
    float y_ = y_z * RAY_SS;
    float z_ = z_y * RAY_SS;

    // Lets init some values
    int vol_x, vol_y, vol_z;
    int volume_index;
    for (int step = 20; step < RAY_STEPS; step++) {
        if (rayptr[index].full) {
            break;
        }

        x = ray.origin.x + x_ * step;
        y = ray.origin.y + y_ * step;
        z = ray.origin.z + z_ * step;
        vol_x = (int)x + vol_x_range / 2;
        vol_y = (int)y + vol_y_range / 2;
        vol_z = (int)z + vol_z_range / 2;
                
        if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && // Only proceed if coordinate is within volume!
            vol_x < vol_x_range && vol_y < vol_y_range && vol_z < vol_z_range) {
            
            volume_index = vol_z * VOL_X * VOL_Y + vol_y * VOL_X + vol_x;

            if (blocks[volume_index].air)
                continue;
            else {
                rayptr[index].acc_color += blocks[volume_index].value;
                rayptr[index].full = true;
            }    
        }
    }
}


CudaOperator::CudaOperator(){
    cudaMallocManaged(&rayptr, NUM_RAYS * sizeof(Ray));
    cudaMallocManaged(&blocks, VOL_X*VOL_Y*VOL_Z*sizeof(Block));
    cout << "Cuda initialized" << endl;
    }


void CudaOperator::newVolume(Block* bs) { 
    cudaMemcpy(blocks, bs, VOL_X*VOL_Y*VOL_Z * sizeof(Block), cudaMemcpyHostToDevice);
    //Volume only needs to go one way as it is not altered.
}


void CudaOperator::rayStep(Ray *rp) {
    time_t start, finish;

    time(&start);
    cudaMemcpy(rayptr, rp, NUM_RAYS * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    time(&finish);
    float t1 = difftime(finish, start);

    time(&start);
    stepKernel << <RAYS_PER_DIM, RAYS_PER_DIM >> > (rayptr, blocks);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();
    time(&finish);
    float t2 = difftime(finish, start);
    
    

    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    time(&start);
    cudaMemcpy(rp, rayptr, NUM_RAYS * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    time(&finish);
    float t3 = difftime(finish, start);
    printf("CopyD time: %f   Step time: %f   CopyH time: %f", t1, t2, t3);
    cout << t2 << endl;
}


