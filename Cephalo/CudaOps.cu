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
    rayptr[index].acc_color = 0;
    rayptr[index].acc_alpha = 0;
    rayptr[index].full = false;


    float sin_pitch = sin(rayptr[index].cam_pitch);
    float cos_pitch = cos(rayptr[index].cam_pitch);
    float sin_yaw = sin(rayptr[index].cam_yaw);
    float cos_yaw = cos(rayptr[index].cam_yaw);

    float x = rayptr[index].rel_unit_vector.x;
    float y = rayptr[index].rel_unit_vector.y;
    float z = rayptr[index].rel_unit_vector.z;

    float x_y = cos_pitch * x + sin_pitch * z;
    float y_y = y;
    float z_y = -sin_pitch * x + cos_pitch * z;

    // Rotate relative vector about z
    float x_z = cos_yaw * x_y - sin_yaw * y_y;
    float y_z = sin_yaw * x_y + cos_yaw * y_y;
    float z_z = z_y;
    
    float x_ = x_z * RAY_SS;
    float y_ = y_z * RAY_SS;
    float z_ = z_z * RAY_SS;

    for (int step = 20; step < RAY_STEPS; step++) {
        if (rayptr[index].full) {
            break;
        }

        float x = rayptr[index].origin.x + x_ * step;
        float y = rayptr[index].origin.y + y_ * step;
        float z = rayptr[index].origin.z + z_ * step;
        int vol_x = (int)x + vol_x_range / 2;
        int vol_y = (int)y + vol_y_range / 2;
        int vol_z = (int)z + vol_z_range / 2;
                
        if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && // Only proceed if coordinate is within volume!
            vol_x < vol_x_range && vol_y < vol_y_range && vol_z < vol_z_range) {
            int volume_index = vol_z * VOL_X * VOL_Y + vol_y * VOL_X + vol_x;
            if (blocks[volume_index].air)
                continue;
            else {
                rayptr[index].acc_color += blocks[volume_index].value;
                rayptr[index].full = true;
            }
            //rayptr[index].acc_alpha += blocks[volume_index].alpha;
     
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
    cudaMemcpy(rayptr, rp, NUM_RAYS * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    stepKernel << <RAYS_PER_DIM, RAYS_PER_DIM >> > (rayptr, blocks);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();


    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(rp, rayptr, NUM_RAYS * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
}


