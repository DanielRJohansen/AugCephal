#include "CudaOps.cuh"

#define vol_x_range VOL_X
#define vol_y_range VOL_Y
#define vol_z_range VOL_Z
#define RAY_SS 0.8
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);


__device__ bool isInVolume(int x, int y, int z) {
    return x >= 0 && y >= 0 && z >= 0 && x < vol_x_range&& y < vol_y_range&& z < vol_z_range;
}


__device__ float lightSeeker(Block* volume, CudaFloat3 pos) {
    float spread = 0.5;
    float brightness = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; y++) {
            for (int z = 1; z < 5; z *= 2) {
                int vol_x = pos.x - spread + spread * x;
                int vol_y = pos.y - spread + spread * y;
                int vol_z = pos.z + z;
                if (isInVolume(vol_x, vol_y, vol_z)) {
                    //if (volume[])
                }
            }
        }
    }
}

__device__ CudaFloat3 makeUnitVector(Ray* ray, CompactCam cc) {
    float x = ray->rel_unit_vector.x;
    float y = ray->rel_unit_vector.y;
    float z = ray->rel_unit_vector.z;


    // Rotate rel vector around y
    float x_y = cc.cos_pitch * x + cc.sin_pitch * z;
    float z_y = -cc.sin_pitch * x + cc.cos_pitch * z;

    // Rotate relative vector about z
    float x_z = cc.cos_yaw * x_y - cc.sin_yaw * y;
    float y_z = cc.sin_yaw * x_y + cc.cos_yaw * y;

    return CudaFloat3(x_z, y_z, z_y);
}
__global__ void stepKernelMS(Ray* rayptr, Block* blocks, CompactCam cc, int offset, uint8_t* image, bool* empty_y_slices, bool* empty_x_slices) {
    // 30 ms
    int index = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x+offset;
    Ray ray = rayptr[index];    // This operation alone takes ~60 ms
    // 90 ms


    //CudaFloat3 unit_vector(x_z, y_z, z_y);
    CudaFloat3 unit_vector = makeUnitVector(&ray, cc);
    CudaRay cray(unit_vector * RAY_SS);
    
    // 110 ms

    Block* cached_block;
    cached_block = &blocks[0];  // Init Block, doesn't matter is never used before another is loaded.
    int prev_vol_index = -1;    // Impossible index
    
    // 140 ms

    for (int step = 20; step < RAY_STEPS; step++) {

        int x = cc.origin.x + cray.step_vector.x * step;
        int y = cc.origin.y + cray.step_vector.y * step;
        int z = cc.origin.z + cray.step_vector.z * step;

        int vol_x = (int)x + vol_x_range / 2;
        int vol_y = (int)y + vol_y_range / 2;
        int vol_z = (int)z + vol_z_range / 2;

        if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && // Only proceed if coordinate is within volume!
            vol_x < vol_x_range && vol_y < vol_y_range && vol_z < vol_z_range) {
            int volume_index = vol_z * VOL_X * VOL_Y + vol_y * VOL_X + vol_x;

            if (vol_z == 0) {
                cray.color.r = 0;
                cray.color.g = 114;
                cray.color.b = 158;
                break;
            }
            else if (empty_y_slices[vol_y] || empty_x_slices[vol_x]) { continue; }

            if (volume_index == prev_vol_index) {
                if (cached_block->ignore) { continue; }
            }
            else {
                prev_vol_index = volume_index;
                if (blocks[volume_index].ignore) { continue; } 
                else { cached_block = &blocks[volume_index]; }
            }

            
            CudaColor block_color = CudaColor(cached_block->color.r, cached_block->color.g, cached_block->color.b);
            cray.color.add(block_color * cached_block->alpha);
            cray.alpha += cached_block->alpha;
            if (cray.alpha >= 1)
                break;
        }
    }
    image[index * 4 + 0] = cray.color.r;
    image[index * 4 + 1] = cray.color.g;
    image[index * 4 + 2] = cray.color.b;
    image[index * 4 + 3] = 255;
}


__global__ void medianFilterKernel(Block* original, Block* volume, circularWindow* windows, int* finished) {
    
    int y = blockIdx.x;  //This fucks shit up if RPD > 1024!!
    int x = threadIdx.x;
    if (x == 0 || y == 0 || x == VOL_X - 1 || y == VOL_Y - 1)
        return;
    int id = y * VOL_X + x;
    
    circularWindow window;

    for (int z = 0; z < 2; z++) {
        for (int yoff = -1; yoff < 2; yoff++) {
            for (int xoff = -1; xoff < 2; xoff++) {
                int vol_index = z * VOL_Y * VOL_X + (y+yoff) * VOL_X + x+xoff;
                window.add(original[vol_index].value);
            }
        }
    }

    for (int z = 1; z < VOL_Z - 1; z++) {
        int vol_index = z * VOL_Y * VOL_X + y * VOL_X + x;
        for (int yoff = -1; yoff < 2; yoff++) {
            for (int xoff = -1; xoff < 2; xoff++) {
                int vol_index_window = (z + 1) * VOL_Y * VOL_X + (y + yoff) * VOL_X + x + xoff;
                window.add(original[vol_index_window].value);
            }
        }            
        volume[vol_index].value = window.step();
    }
    
    *finished = 1;
}

CudaOperator::CudaOperator(){
    cudaMallocManaged(&rayptr, NUM_RAYS * sizeof(Ray));
    cudaMallocManaged(&blocks, VOL_X*VOL_Y*VOL_Z*sizeof(Block));
    cudaMallocManaged(&ray_block, 4 * sizeof(Float2));
    cudaMallocManaged(&compact_cam, sizeof(CompactCam));
    cudaMallocManaged(&dev_image, NUM_RAYS * 4 * sizeof(uint8_t));
    cudaMallocManaged(&dev_empty_y_slices, VOL_Y * N_STREAMS * sizeof(bool));
    cudaMallocManaged(&dev_empty_x_slices, VOL_Y * N_STREAMS * sizeof(bool));
    host_image = new uint8_t[NUM_RAYS * 4];

    for (int y = 0; y < RAY_BLOCKS_PER_DIM; y++) {
        for (int x = 0; x < RAY_BLOCKS_PER_DIM; x++) {
            ray_block[y * RAY_BLOCKS_PER_DIM + x] = Float2(x, y);
        }
    }
    blocks_per_sm = NUM_RAYS / (THREADS_PER_BLOCK * N_STREAMS);
    stream_size = blocks_per_sm * THREADS_PER_BLOCK;
    ray_stream_bytes = stream_size * sizeof(Ray);
    image_stream_bytes = stream_size * 4 * sizeof(uint8_t);
    printf("Blocks per SM: %d \n", blocks_per_sm);

    cout << "Cuda initialized" << endl;
    }


void CudaOperator::newVolume(Block* bs) { 
    cudaMemcpy(blocks, bs, VOL_X*VOL_Y*VOL_Z * sizeof(Block), cudaMemcpyHostToDevice);
}
void CudaOperator::updateEmptySlices(bool* y_empty, bool* x_empty) {
    bool* host_empty_y_slices = new bool[VOL_Y];
    bool* host_empty_x_slices = new bool[VOL_X];
    for (int y = 0; y < VOL_Y; y++) {
        host_empty_y_slices[y] = y_empty[y];
    }
    for (int x = 0; x < VOL_X; x++) {
        host_empty_x_slices[x] = x_empty[x];
    }
    cudaMemcpy(dev_empty_y_slices, host_empty_y_slices,  VOL_Y * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_empty_x_slices, host_empty_x_slices, VOL_X * sizeof(bool), cudaMemcpyHostToDevice);
}

void CudaOperator::rayStepMS(Ray* rp, CompactCam cc, sf::Texture* texture) {
    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&(stream[i]));
    }


    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&rayptr[offset], &rp[offset], ray_stream_bytes, cudaMemcpyHostToDevice, stream[i]);
    }


    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        stepKernelMS << <blocks_per_sm, THREADS_PER_BLOCK, 0, stream[i] >> > (rayptr,
            blocks, cc, offset, dev_image, dev_empty_y_slices, dev_empty_x_slices);
    }
    printf("Rendering...");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&host_image[offset*4], &dev_image[offset*4], image_stream_bytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    printf("  Received!\n");
    //cudaDeviceSynchronize();
    texture->update(host_image);

    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }
}


void CudaOperator::medianFilter(Block* ori, Block* vol) {
    int num_blocks = VOL_X * VOL_Y * VOL_Z;

    cout << "Starting median filter of kernel size 3" << endl;
    cout << "Requires space " << (VOL_X*VOL_Y * sizeof(circularWindow) + 2*num_blocks*sizeof(Block))/ 1000000 << " Mb" << endl;
    Block* original; Block* volume;
    circularWindow* windows;
    int* finished = new int;
    *finished = 0;
    cudaMallocManaged(&finished, sizeof(int));
    cudaMallocManaged(&original, num_blocks * sizeof(Block));
    cudaMallocManaged(&volume, num_blocks * sizeof(Block));
    cudaMallocManaged(&windows, VOL_X*VOL_Y * sizeof(circularWindow));

    // Hacky workaround to move the objects onto the device.
    circularWindow* cc;
    cc = new circularWindow[num_blocks];
    cudaMemcpy(windows, cc, num_blocks * sizeof(circularWindow), cudaMemcpyHostToDevice);


   


    cudaMemcpy(original, ori, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaMemcpy(volume, vol, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    medianFilterKernel << <VOL_Y, VOL_X >> > (original, volume, windows, finished);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();

    cout << "Finished " << *finished << endl;
    
    
    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(vol, volume, num_blocks * sizeof(Block), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(original);
    cudaFree(volume);
    cudaFree(windows);
}






__device__ void circularWindow::add(float val) {  
    window[head] = val;
    head++;
    if (head == 27) 
        head = 0;
}
__device__ void circularWindow::sortWindow() {
    copyWindow();
    int lowest_index = 0;
    for (int i = 0; i < size; i++) {
        float lowest = 99999;
        for (int j = 0; j < size; j++) {
            if (window_copy[j] < lowest) {
                lowest = window_copy[j];
                lowest_index = j;
            }
        }
        window_sorted[i] = window_copy[lowest_index];
        window_copy[lowest_index] = 99999;
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