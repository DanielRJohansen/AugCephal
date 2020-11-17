#include "CudaOps.cuh"

#define vol_x_range VOL_X
#define vol_y_range VOL_Y
#define vol_z_range VOL_Z
#define RAY_SS 1
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void stepKernel(Ray* rayptr, Block *blocks, Float2* ray_block) {
    int ray_x = threadIdx.x + 1024 * ray_block->x;
    int ray_y = blockIdx.x + 1024 * ray_block->y;
    int index = ray_y * RAYS_PER_DIM + ray_x; 

    //Reset ray
    Ray ray = rayptr[index];
    rayptr[index].color.r = 0;
    rayptr[index].color.g = 0;
    rayptr[index].color.b = 0;
    rayptr[index].alpha = 0;
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
            //Block block = blocks[volume_index];
            //if (blocks[volume_index].air || blocks[volume_index].bone || blocks[volume_index].soft_tissue || blocks[volume_index].fat)
            if (blocks[volume_index].ignore)
                continue; 
            else {
                rayptr[index].color.r += blocks[volume_index].color.r * blocks[volume_index].alpha;
                rayptr[index].color.g += blocks[volume_index].color.g * blocks[volume_index].alpha;
                rayptr[index].color.b += blocks[volume_index].color.b * blocks[volume_index].alpha;

                //rayptr[index].acc_color += blocks[volume_index].value * blocks[volume_index].alpha;
                rayptr[index].alpha += blocks[volume_index].alpha;
                if (rayptr[index].alpha >= 1)
                    rayptr[index].full = true;
            }    
        }
    }
}

__global__ void testKernel(Ray* rayptr, int a) {
    int b = 0;
    return;
}

__global__ void stepKernelMS(Ray* rayptr, Block* blocks, int offset) {
  
    int index = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x+offset;
    //Reset ray
    Ray ray = rayptr[index];
    rayptr[index].color.r = 0;
    rayptr[index].color.g = 0;
    rayptr[index].color.b = 0;
    rayptr[index].alpha = 0;
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
            //Block block = blocks[volume_index];
            //if (blocks[volume_index].air || blocks[volume_index].bone || blocks[volume_index].soft_tissue || blocks[volume_index].fat)
            if (blocks[volume_index].ignore)
                continue;
            else {
                rayptr[index].color.r += blocks[volume_index].color.r * blocks[volume_index].alpha;
                rayptr[index].color.g += blocks[volume_index].color.g * blocks[volume_index].alpha;
                rayptr[index].color.b += blocks[volume_index].color.b * blocks[volume_index].alpha;

                //rayptr[index].acc_color += blocks[volume_index].value * blocks[volume_index].alpha;
                rayptr[index].alpha += blocks[volume_index].alpha;
                if (rayptr[index].alpha >= 1)
                    rayptr[index].full = true;
            }
        }
    }
}


__global__ void medianFilterKernel(Block* original, Block* volume, circularWindow* windows, int* finished) {
    
    int y = blockIdx.x;  //This fucks shit up if RPD > 1024!!
    int x = threadIdx.x;
    int id = y * VOL_X + x;
    
    circularWindow window = windows[id];

    if (y * x != 0 && x < VOL_X-1 && y < VOL_Y-1) {   //Can't do this if on the edge
        for (int z = 0; z < 2; z++) {

            for (int yoff = -1; yoff < 2; yoff++) {
                for (int xoff = -1; xoff < 2; xoff++) {
                    int vol_index = z * VOL_Y * VOL_X + (y+yoff) * VOL_X + x+xoff;
                    window.add(original[vol_index].value);
                }
            }
        }
    }
    
    
    for (int z = 2; z < VOL_Z - 1; z++) {
        int vol_index = z * VOL_Y * VOL_X + y * VOL_X + x;

        // If any is 0, we are on the edge
        if (x*y*z == 0 || x == (VOL_X-1) || y == (VOL_Y - 1) || z == (VOL_Z-1)) {
            //volume[vol_index].air = true;
        }
        else {  // Copy top
            for (int yoff = -1; yoff < 2; yoff++) {
                for (int xoff = -1; xoff < 2; xoff++) {
                    int vol_index_window = (z + 1) * VOL_Y * VOL_X + (y + yoff) * VOL_X + x + xoff;
                    window.add(original[vol_index_window].value);
                }
            }
            
            volume[vol_index].value = window.step();
            //volume[vol_index].value = window.step();
        }
    }
    
    finished[1] = 1;
}

CudaOperator::CudaOperator(){
    cudaMallocManaged(&rayptr, NUM_RAYS * sizeof(Ray));
    cudaMallocManaged(&blocks, VOL_X*VOL_Y*VOL_Z*sizeof(Block));
    cudaMallocManaged(&ray_block, 4 * sizeof(Float2));
    for (int y = 0; y < RAY_BLOCKS_PER_DIM; y++) {
        for (int x = 0; x < RAY_BLOCKS_PER_DIM; x++) {
            ray_block[y * RAY_BLOCKS_PER_DIM + x] = Float2(x, y);
        }
    }
    blocks_per_sm = NUM_RAYS / (THREADS_PER_BLOCK * N_STREAMS);
    stream_size = blocks_per_sm * THREADS_PER_BLOCK;
    stream_bytes = stream_size * sizeof(Ray);
    printf("Blocks per SM: %d \n", blocks_per_sm);

    cout << "Cuda initialized" << endl;
    }



void CudaOperator::newVolume(Block* bs) { 
    cudaMemcpy(blocks, bs, VOL_X*VOL_Y*VOL_Z * sizeof(Block), cudaMemcpyHostToDevice);
    //Volume only needs to go one way as it is not altered.
}


void CudaOperator::rayStep(Ray *rp) {
    // Copy rayptr to device
    cudaMemcpy(rayptr, rp, NUM_RAYS * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    stepKernel << <1024, 1024 >> > (rayptr, blocks, &ray_block[0]);    // RPD blocks (y), RPD threads(x)
    //stepKernel << <1024, 1024 >> > (rayptr, blocks, &ray_block[1]);
    //stepKernel << <1024, 1024 >> > (rayptr, blocks, &ray_block[2]);
    //stepKernel << <1024, 1024 >> > (rayptr, blocks, &ray_block[3]);

    cudaDeviceSynchronize();

    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(rp, rayptr, NUM_RAYS * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void CudaOperator::rayStepMS(Ray* rp) {
    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&(stream[i]));
    }
    //printf("Sending\n");
    
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&rayptr[offset], &rp[offset], stream_bytes, cudaMemcpyHostToDevice, stream[i]);
    }
    //printf("to device\n");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        stepKernelMS << <blocks_per_sm, THREADS_PER_BLOCK, 0, stream[i] >> > (rayptr, blocks, offset);

    }
    //printf("execution");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&rp[offset], &rayptr[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[i]);

    }

    //printf("Received\n");
    cudaDeviceSynchronize();
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }
}

void CudaOperator::medianFilter(Block* ori, Block* vol) {
    int num_blocks = VOL_X * VOL_Y * VOL_Z;

    cout << "Starting median filter of kernel size 3" << endl;
    cout << "Requires space " << num_blocks * sizeof(circularWindow) / 1000000 << " Mb" << endl;
    Block* original; Block* volume;
    circularWindow* windows;
    int* finished = new int[2];
    cudaMallocManaged(&finished, sizeof(int));
    cudaMallocManaged(&original, num_blocks * sizeof(Block));
    cudaMallocManaged(&volume, num_blocks * sizeof(Block));
    cudaMallocManaged(&windows, num_blocks * sizeof(circularWindow));

    // Hacky workaround to move the objects onto the device.
    circularWindow* cc;
    cc = new circularWindow[num_blocks];
    cudaMemcpy(windows, cc, num_blocks * sizeof(circularWindow), cudaMemcpyHostToDevice);


   


    cudaMemcpy(original, ori, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaMemcpy(volume, vol, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    medianFilterKernel << <VOL_Y, VOL_X >> > (original, volume, windows, finished);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();

    cout << "Finished " << finished[1] << endl;
    
    
    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(vol, volume, num_blocks * sizeof(Block), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(original);
    cudaFree(volume);
    cudaFree(windows);
}


/*__device__ circularWindow::circularWindow() {
    window = new float[size];
    window_copy = new float[size];
    window_sorted = new float[size];
}*/
__device__ void circularWindow::add(float val) {  
    window[head] = val;
    head++;
    if (head == 27) // Doesn't work if i put size, noooooo idea :/
        head = 0;
}
__device__ void circularWindow::sortWindow() {
    copyWindow();
    int lowest_index = 0;
    for (int i = 0; i < size; i++) {
        float lowest = 9999;
        for (int j = 0; j < size; j++) {
            if (window_copy[j] < lowest) {
                lowest = window_copy[j];
                lowest_index = j;
            }
        }
        window_sorted[i] = window_copy[lowest_index];
        window_copy[lowest_index] = 9999;
    }
}
__device__ void circularWindow::copyWindow() {
    for (int i = 0; i < size; i++) {
        window_copy[i] = window[i];
    }
}
__device__ float circularWindow::step() {
    sortWindow(); 
    return window_sorted[14];
}