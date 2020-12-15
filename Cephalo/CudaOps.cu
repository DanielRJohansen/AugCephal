#include "CudaOps.cuh"

#define vol_x_range VOL_X
#define vol_y_range VOL_Y
#define vol_z_range VOL_Z
#define RAY_SS 0.6
#define e 2.71828

#define outside_spectrum OUTSIDE_SPECTRUM

__device__ const float TOO_FEW = -2000;
__device__ const float ERASE = -2001;
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);


__device__ bool isInVolume(int x, int y, int z) {
    return x >= 0 && y >= 0 && z >= 0 && x < vol_x_range&& y < vol_y_range&& z < vol_z_range;
}

__device__ int xyzToIndex(int vol_x, int vol_y, int vol_z) {
    return vol_z * VOL_X * VOL_Y + vol_y * VOL_X + vol_x;
}

__device__ float activationFunction(float counts) {
    return 2 / (1 + powf(e, (-counts / 4.))) - 1.;
}

__device__ float lightSeeker(Block* volume, CudaFloat3 pos) {
    float spread = 1;
    float brightness = 1;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {

            // Upward seeking
            brightness += 1;
            for (int z = 1; z <= 64; z *= 2) {
                int vol_x = pos.x - spread + spread * x;
                int vol_y = pos.y - spread + spread * y;
                int vol_z = pos.z + z;
                if (isInVolume(vol_x, vol_y, vol_z)) {
                    int index = xyzToIndex(vol_x, vol_y, vol_z);
                    if (!volume[index].ignore) { brightness -= 1; break; }
                    //else { break; }
                }
                //else { brightness += 1; }
            }
        }
    }
    return activationFunction(brightness);
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

    for (int step = 500; step < RAY_STEPS; step++) {

        int x = cc.origin.x + cray.step_vector.x * step;
        int y = cc.origin.y + cray.step_vector.y * step;
        int z = cc.origin.z + cray.step_vector.z * step;

        int vol_x = (int)x + vol_x_range / 2;
        int vol_y = (int)y + vol_y_range / 2;
        int vol_z = (int)z + vol_z_range / 2;

        if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && // Only proceed if coordinate is within volume!
            vol_x < vol_x_range && vol_y < vol_y_range && vol_z < vol_z_range) {
            int volume_index = xyzToIndex(vol_x, vol_y, vol_z);

            /*if (vol_z == 0) {
                cray.color.r = 0;
                cray.color.g = 114;
                cray.color.b = 158;
                break;
            }*/
            if (empty_y_slices[vol_y] || empty_x_slices[vol_x]) { continue; }

            if (volume_index == prev_vol_index) {
                if (cached_block->ignore) { continue; }
            }
            else {
                prev_vol_index = volume_index;
                if (blocks[volume_index].ignore) { continue; } 
                else { cached_block = &blocks[volume_index]; }
            }

            
            CudaColor block_color = CudaColor(cached_block->color.r, cached_block->color.g, cached_block->color.b);
            float brightness = lightSeeker(blocks, CudaFloat3(vol_x, vol_y, vol_z));
            block_color = block_color * brightness;
            cray.color.add(block_color * cached_block->alpha);
            cray.alpha += cached_block->alpha;
            if (cray.alpha >= 1)
                break;
        }
    }
    cray.color.cap();   //Caps each channel at 255
    image[index * 4 + 0] = (int) cray.color.r;
    image[index * 4 + 1] = (int) cray.color.g;
    image[index * 4 + 2] = (int) cray.color.b;
    image[index * 4 + 3] = 255;
}


__global__ void rotatingMaskFilterKernel(Block* original, Block* volume, int ignorelow, int ignorehigh ) {
    int y = blockIdx.x;  //This fucks shit up if RPD > 1024!!
    int x = threadIdx.x;
    if (x < 2 || y < 2 || x > VOL_X - 3 || y > VOL_Y - 3)
        return;
    
    const int num_masks = 27;
    const int kernel_size = 5 * 5 * 5;
    
    CudaMask masks[27];

    // Initialize masks
    int i = 0;
    for (int zs = 0; zs < 3; zs++) {
        for (int ys = 0; ys < 3; ys++) {
            for (int xs = 0; xs < 3; xs++) {
                masks[i] = CudaMask(xs, ys, zs);
                i++;
            }
        }
    }

    for (int z = 0; z < VOL_Z; z++) {
        Block block = original[xyzToIndex(x, y, z)];
        if (block.hu_val < ignorelow || block.hu_val > ignorehigh || block.ignore)       // as to not erase bone or brigthen air
            continue;

        // Generate kernel
        float kernel[5*5*5];
        int i = 0;
        for (int z_ = z - 2; z_ <= z + 2; z_++) {
            for (int y_ = y - 2; y_ <= y + 2; y_++) {
                for (int x_ = x - 2; x_ <= x + 2; x_++) {
                    //kernel[i] = original[xyzToIndex(x_, y_, z_)].hu_val;
                    if (isInVolume(x_, y_, z_))
                        kernel[i] = original[xyzToIndex(x_, y_, z_)].hu_val;
                    else
                        kernel[i] =  EMPTYBLOCK;
                    i++;
                }
            }
        }
        float best_mean = 0;
        float lowest_var = 999999;
        float kernel_copy[5*5*5];
        for (int i = 0; i < num_masks; i++) {
            for (int j = 0; j < kernel_size; j++)
                kernel_copy[j] = kernel[j];
            float mean = masks[i].applyMask(kernel_copy);
            float var = masks[i].calcVar(kernel_copy, mean);
            if (var < lowest_var) {
                lowest_var = var;
                best_mean = mean;
            }
        }
        volume[xyzToIndex(x, y, z)].hu_val = best_mean;
    } 
}
__global__ void kMeansKernel(Block* volume, CudaCluster* clusters, int num_clusters, int operation) {   //operation init, iterate, assign (0,1,2)
    int y = blockIdx.x;  //This fucks shit up if RPD > 1024!!
    int x = threadIdx.x;


    if (operation == 0) {           // Initialize clusters to pseudo random values
        for (int z = 0; z < VOL_Z; z++) {
            if (volume[xyzToIndex(x, y, z)].ignore)
                continue;
            int pseudorandom_cluster_index = (z * VOL_Y * VOL_X + y * VOL_X + x) % num_clusters;
            clusters[pseudorandom_cluster_index].addMember(volume[xyzToIndex(x, y, z)].hu_val);
        }
    }
    else if (operation == 1) {        // Iterate all values
        for (int z = 0; z < VOL_Z; z++) {
            if (volume[xyzToIndex(x, y, z)].ignore)
                continue;
            float hu_val = volume[xyzToIndex(x, y, z)].hu_val;
            int best_index = 0;
            float shortest_dist = 99999;
            for (int i = 0; i < num_clusters; i++) {
                float dist = clusters[i].belongingScore(hu_val);
                if (dist < shortest_dist) {
                    shortest_dist = dist;
                    best_index = i;
                }
            }
            clusters[best_index].addMember(hu_val);
        }       //Update cluster MUST be called from HOST (else 250k commands will be sent...)
    }
    else if (operation == 2) {
        for (int z = 0; z < VOL_Z; z++) {
            if (volume[xyzToIndex(x, y, z)].ignore)
                continue;
            float hu_val = volume[xyzToIndex(x, y, z)].hu_val;
            int best_index = 0;
            float shortest_dist = 99999;
            for (int i = 0; i < num_clusters; i++) {
                float dist = clusters[i].belongingScore(hu_val);
                if (dist < shortest_dist) {
                    shortest_dist = dist;
                    best_index = i;
                }
            }
            volume[xyzToIndex(x, y, z)].hu_val = clusters[best_index].getClusterMean();
        }
    }
}

__global__ void medianFilterKernel(Block* original, Block* volume, int* finished) {
    
    int y = blockIdx.x;  //This fucks shit up if RPD > 1024!!
    int x = threadIdx.x;
    if (x == 0 || y == 0 || x == VOL_X - 1 || y == VOL_Y - 1)
        return;
    
    circularWindow window;

    for (int z = 0; z < 2; z++) {
        for (int yoff = -1; yoff < 2; yoff++) {
            for (int xoff = -1; xoff < 2; xoff++) {
                int vol_index = z * VOL_Y * VOL_X + (y+yoff) * VOL_X + x+xoff;
                window.add(original[vol_index].hu_val);
            }
        }
    }

    for (int z = 1; z < VOL_Z - 1; z++) {
        //volume[i].ignore = true;
        int vol_index = z * VOL_Y * VOL_X + y * VOL_X + x;
        for (int yoff = -1; yoff < 2; yoff++) {
            for (int xoff = -1; xoff < 2; xoff++) {
                int vol_index_window = (z + 1) * VOL_Y * VOL_X + (y + yoff) * VOL_X + x + xoff;
                window.add(original[vol_index_window].hu_val);
            }
        }
        // It is important we add the upper plane, even when the block is to be ignored!
        if (volume[vol_index].ignore) { continue; } 
        int median_val = window.step();
        if (median_val == ERASE) { volume[vol_index].ignore = true; }
        else if (median_val != TOO_FEW) // Otherwise dont change the value
            volume[vol_index].hu_val = (volume[vol_index].hu_val + median_val) / 2.;
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
    auto start = chrono::high_resolution_clock::now();


    cout << "Starting median filter of kernel size 3" << endl;
    cout << "Requires space " << (VOL_X*VOL_Y * sizeof(circularWindow) + 2*num_blocks*sizeof(Block))/ 1000000 << " Mb" << endl;
    Block* original; Block* volume;
    int* finished = new int;
    *finished = 0;
    cudaMallocManaged(&finished, sizeof(int));
    cudaMallocManaged(&original, num_blocks * sizeof(Block));
    cudaMallocManaged(&volume, num_blocks * sizeof(Block));



    cudaMemcpy(original, ori, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaMemcpy(volume, vol, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    medianFilterKernel << <VOL_Y, VOL_X >> > (original, volume, finished);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();


    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Finished " << *finished;
    printf("   Filter applied in %d ms.\n", duration);
    
    
    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(vol, volume, num_blocks * sizeof(Block), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(original);
    cudaFree(volume);
}

void CudaOperator::rotatingMaskFilter(Block* ori, Block* vol) {
    int num_blocks = VOL_X * VOL_Y * VOL_Z;
    auto start = chrono::high_resolution_clock::now();


    cout << "Starting rotating mask filter of kernel size 3" << endl;
    cout << "Requires space " << (VOL_X * VOL_Y * 125 * sizeof(CudaMask) + 2 * num_blocks * sizeof(Block)) / 1000000 << " Mb" << endl;
    Block* original; Block* volume;

    cudaMallocManaged(&original, num_blocks * sizeof(Block));
    cudaMallocManaged(&volume, num_blocks * sizeof(Block));



    cudaMemcpy(original, ori, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaMemcpy(volume, vol, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    rotatingMaskFilterKernel << <VOL_Y, VOL_X >> > (original, volume, -700, 800);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();


    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    printf("  Rotating Mask Filter applied in %d ms.\n", duration);


    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(vol, volume, num_blocks * sizeof(Block), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(original);
    cudaFree(volume);
}

void CudaOperator::kMeansClustering(Block* vol) {
    int num_blocks = VOL_X * VOL_Y * VOL_Z;
    auto start = chrono::high_resolution_clock::now();

    printf("Starting k-means with %d clusters\n", num_K);
    printf("Requires space: %d + %d MB (Clusters, Volume)\n", num_K * sizeof(CudaCluster) / 1000000, num_blocks * sizeof(Block) / 1000000);

    Block* volume;
    CudaCluster* clusters;
    cudaMallocManaged(&volume, num_blocks * sizeof(Block));
    cudaMallocManaged(&clusters, num_K * sizeof(CudaCluster));

    cudaMemcpy(volume, vol, num_blocks * sizeof(Block), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    kMeansKernel << <VOL_Y, VOL_X >> > (volume, clusters, num_K, 0);    // Init
    cudaDeviceSynchronize();
    for (int i = 0; i < num_K; i++) { clusters[i].updateCluster();}
    for (int iter = 0; iter < 10; iter++) {
        kMeansKernel << <VOL_Y, VOL_X >> > (volume, clusters, num_K, 1);    // Iter
        cudaDeviceSynchronize();
        for (int i = 0; i < num_K; i++) { clusters[i].updateCluster(); printf("%f  \n", clusters[i].mean); }
        printf("\n");
    }  
    kMeansKernel << <VOL_Y, VOL_X >> > (volume, clusters, num_K, 2);    // Assign
    cudaDeviceSynchronize();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    printf(" K-means with %d iterations applied in %d ms.\n", km_iterations, duration);


    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(vol, volume, num_blocks * sizeof(Block), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(volume);
}




__device__ void circularWindow::add(int val) {  
    window[head] = val;
    head++;
    if (head == WINDOW_SIZE)
        head = 0;
}
__device__ void circularWindow::sortWindow() {
    copyWindow();
    int lowest_index = 0;
    int max = 99999;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float lowest = max;
        for (int j = 0; j < WINDOW_SIZE; j++) {
            if (window_copy[j] < lowest) {
                lowest = window_copy[j];
                lowest_index = j;
            }
        }
        window_sorted[i] = window_copy[lowest_index];
        window_copy[lowest_index] = max;
    }
}
__device__ void circularWindow::copyWindow() {
    for (int i = 0; i < WINDOW_SIZE; i++) {
        window_copy[i] = window[i];
    }
}
__device__ int circularWindow::numOutsideSpectrum() {
    int num = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        if (window_sorted[i] == outside_spectrum) { num++; }
        else { return num; }
    }
    return num;
}
__device__ int circularWindow::step() {
    sortWindow(); 

    int num_os = numOutsideSpectrum();
    //return ERASE;
    if (num_os > 25) { return ERASE; }
    if (num_os > 20) { return TOO_FEW; }
    return window_sorted[13 + num_os/2];
}