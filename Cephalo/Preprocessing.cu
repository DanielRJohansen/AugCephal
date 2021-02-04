#include "Preprocessing.cuh"

void checkCudaError() {
    cudaError_t err = cudaGetLastError();        // Get error code
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}








__device__ __host__ int xyzToIndex2(Int3 coord, Int3 size) {
    return coord.z * size.y * size.x + coord.y * size.x + coord.x;
}


void Preprocessor::insertImInVolume(cv::Mat img, int z) {
    for (int y = 0; y < input_size.y; y++) {
        for (int x = 0; x < input_size.x; x++) {
            float hu = img.at<uint16_t>(y, x) - 32768.; // Load as float here, as we do all further calcs on floats in GPU
            raw_scan[xyzToIndex2(Int3(x,y,z), input_size)] = hu;
        }
    }
}

void Preprocessor::loadScans(string folder_path) {
    int successes = 0;
    stringvec v;
    printf("Reading directory %s\n", folder_path.c_str());
    read_directory(folder_path, v);
    for (int i = 2; i < input_size.z + 2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        printf("Loading slice: %s               \r", im_path.c_str());

        cv::Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = input_size.z - 1 - i + 2;
        if (img.empty()) {
            cout << "\n        Failed!\n" << endl;
            return;
        }
        else successes++;
        insertImInVolume(img, z);
    }

    printf("\n%d Slices loaded succesfully\n", successes);
}


__global__ void conversionKernel(Voxel* voxels, float* hu_vals, Int3 size) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    for (int z = 0; z < size.z; z++) {
        int index = xyzToIndex2(Int3(x, y, z), size);
        voxels[index].hu_val = 600;
    }
}

Volume* Preprocessor::convertToVolume(float* scan, Int3 size) {
    auto start = chrono::high_resolution_clock::now();
    int len = size.x * size.y * size.z;
    int bytesize = len * sizeof(Voxel);

    // Initialize voxels
    Voxel* v_host = new Voxel[len];
    for (int i = 0; i < len; i++)
        v_host[i].hu_val = scan[i];
    
    // Move voxels to GPU
    Voxel* v_device;
    cudaMallocManaged(&v_device, bytesize);
    cudaMemcpy(v_device, v_host, bytesize, cudaMemcpyHostToDevice);
    printf("%d MB of VRAM allocated to Voxels\n", bytesize / 1000000);

    // C
    Volume* volume = new Volume(v_device, size);

    delete(scan, v_host);

    printf("CPU side conversion in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));
    return volume;
}



__global__ void setIgnoreBelowKernel(Voxel* voxels, float below, Int3 size) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    for (int z = 0; z < size.z; z++) {
        int index = xyzToIndex2(Int3(x, y, z), size);
        if (voxels[index].hu_val < below)
            voxels[index].ignore = true;
    }
}
void Preprocessor::setIgnoreBelow(Volume* volume, float below) {
    Int3 size = volume->size;
    setIgnoreBelowKernel << < size.y, size.x >> > (volume->voxels, below, size);
    cudaDeviceSynchronize();
}

float* makeNormvalCopy(Volume* vol) {
    Voxel* hostvoxels = new Voxel[vol->len];
    cudaMemcpy(hostvoxels, vol->voxels, vol->len * sizeof(Voxel), cudaMemcpyDeviceToHost);
    float* copy_host = new float[vol->len];
    for (int i = 0; i < vol->len; i++)
        copy_host[i] = hostvoxels[i].norm_val;

    float* copynorms;
    cudaMallocManaged(&copynorms, vol->len * sizeof(float));
    cudaMemcpy(copynorms, copy_host, vol->len * sizeof(float), cudaMemcpyHostToDevice);
    return copynorms;
}

__global__ void setColumnIgnoresKernel(Voxel* voxels, bool* xyColumnIgnores, Int3 size) {
    int x = blockIdx.x;
    int y = threadIdx.x; 
    int ignore_index = y * size.x + x;
    xyColumnIgnores[ignore_index] = 0;
    int counts = 0;
    for (int z = 0; z < size.z; z++) {
        int index = xyzToIndex2(Int3(x, y, z), size);
        if (!voxels[index].ignore) {
            return;
        }            
    }
    xyColumnIgnores[ignore_index] = 1;
}

void Preprocessor::setColumnIgnores(Volume* volume) {
    Int3 size = volume->size;
    int column_len = size.x * size.y;
    int boolbytesize = column_len * sizeof(bool);



    
    cudaMallocManaged(&volume->xyColumnIgnores, boolbytesize);

    printf("Allocating xy ignore table of size: %d Kb\n", boolbytesize / 1000);

    setColumnIgnoresKernel << < size.y, size.x >> > (volume->voxels, volume->xyColumnIgnores, size);
    cudaDeviceSynchronize();



    volume->CB = new CompactBool(volume->xyColumnIgnores, column_len);

}




__global__ void colorFromNormvalKernel(Voxel* voxels, Int3 size) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    for (int z = 0; z < size.z; z++) {
        int index = xyzToIndex2(Int3(x, y, z), size);
        voxels[index].color = CudaColor(voxels[index].norm_val);
    }
}

void Preprocessor::colorFromNormval(Volume* volume) {
    Int3 size = volume->size;
    colorFromNormvalKernel << < size.y, size.x >> > (volume->voxels, size);
    cudaDeviceSynchronize();
}














__global__ void windowKernel(Voxel* voxels, float min, float max, Int3 size) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    for (int z = 0; z < size.z; z++) {
        int index = xyzToIndex2(Int3(x, y, z), size);
        //float a = voxels[index].norm_val;
        voxels[index].norm(min, max);
    }
}

void Preprocessor::windowVolume(Volume* volume, float min, float max) {
    auto start = chrono::high_resolution_clock::now();
    Int3 size = volume->size;
    //windowKernel <<< size.y, size.x >>> (gpu_voxels, min, max, size);
    windowKernel << < size.y, size.x >> > (volume->voxels, min, max, size);

    cudaError_t err = cudaGetLastError();        // Get error code
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaDeviceSynchronize();
    printf("Windowing executed in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));
}










void Preprocessor::speedTest() {
    int len = 6000000000;
    char* host = new char[len];
    char* device;    
    cudaMallocManaged(&device, len * sizeof(char));
    auto t1 = chrono::high_resolution_clock::now();
    cudaMemcpy(device, host, len * sizeof(char), cudaMemcpyHostToDevice);
    printf("Sent in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1));

    auto t2 = chrono::high_resolution_clock::now();
    cudaMemcpy(host, device, len * sizeof(char), cudaMemcpyDeviceToHost);
    printf("Received in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t2));

}













__device__ inline bool isInVolume(Int3 coord, Int3 size) {
    return coord.x >= 0 && coord.y >= 0 && coord.z >= 0 && coord.x < size.x&& coord.y < size.y&& coord.z < size.z;
}
__device__ inline int xyzToIndex(Int3 coord, Int3 size) {
    return coord.z * size.y * size.x + coord.y * size.x + coord.x;
}
__host__ __device__ void makeMasks(CudaMask* masks) {
    int i = 0;
    for (int zs = 0; zs < 3; zs++) {
        for (int ys = 0; ys < 3; ys++) {
            for (int xs = 0; xs < 3; xs++) {
                masks[i] = CudaMask(xs, ys, zs);
                i++;
            }
        }
    }
    
    for (int i = 0; i < 3; i++) {
        masks[i] = CudaMask();
    }
    i = 0;
    for (int z = 0; z < 5; z++) {
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                //Flats
                masks[27 + 0].mask[i] = (z == 2);
                masks[27 + 1].mask[i] = (y == 2);
                masks[27 + 2].mask[i] = (x == 2);

                // Pillars
                masks[27 + 3].mask[i] = (z == 2 && y == 2);
                masks[27 + 4].mask[i] = (y == 2 && x == 2);
                masks[27 + 5].mask[i] = (x == 2 && z == 2);

                // Semi crooked pillars
                masks[27 + 6].mask[i] = (z == 2 && x == y);
                masks[27 + 7].mask[i] = (z == 2 && x == -y);
                masks[27 + 8].mask[i] = (y == 2 && z == x);
                masks[27 + 9].mask[i] = (y == 2 && z == -x);
                masks[27 + 10].mask[i] = (x == 2 && z == y);
                masks[27 + 11].mask[i] = (x == 2 && z == -y);

                // Full crooked pillars
                masks[27 + 12].mask[i] = (x == y && y == z);
                masks[27 + 13].mask[i] = (x == y && y == -z);
                masks[27 + 14].mask[i] = (-x == y && y == z);
                masks[27 + 15].mask[i] = (-x == y && y == -z);
                /*masks[27 + 3].mask[i] = (z == 0 && abs(x) - 1 <= 0 && abs(y) - 1 <= 0);
                masks[27 + 4].mask[i] = (y == 0 && abs(x) - 1 <= 0 && abs(z) - 1 <= 0);
                masks[27 + 5].mask[i] = (x == 0 && abs(y) - 1 <= 0 && abs(z) - 1 <= 0);*/

                i++;
            }
        }
    }
}

__device__ void fetchWindow(Voxel* voxelcopy, float* kernel, Int3 pos, Int3 size) {
    int i = 0;
    for (int z_ = pos.z - 2; z_ <= pos.z + 2; z_++) {
        for (int y_ = pos.y - 2; y_ <= pos.y + 2; y_++) {
            for (int x_ = pos.x - 2; x_ <= pos.x + 2; x_++) {
                Int3 pos_ = Int3(x_, y_, z_);
                if (pos.z > 0 && z_ < pos.z + 2)
                    kernel[i] = kernel[i + 25];
                else if (!isInVolume(pos_, size))
                    kernel[i] = OUTSIDEVOL;
                else
                {
                    if (voxelcopy[xyzToIndex(pos_, size)].ignore)
                        kernel[i] = ISIGNORE;
                    else
                        kernel[i] = voxelcopy[xyzToIndex(pos_, size)].norm_val;
                }
                i++;
            }
        }
    }
}

__global__ void rotatingMaskFilterKernel(Voxel* voxels, Voxel* voxelcopy, Int3 size, unsigned* ignores, CudaMask* globalmasks) {
    int y = blockIdx.x;  
    int x = threadIdx.x;
    
    CompactBool CB;
    if (CB.getBit(ignores, y * size.x + x)) {
        voxels[xyzToIndex(Int3(x, y, 0), size)].norm_val = 1;
        return;
    }

    // Initialize masks
    CudaMask masks_init[43];
    CudaMask* masks = masks_init;
    for (int i = 0; i < 43; i++)
        masks[i] = globalmasks[i];

    float kernel_[5 * 5 * 5];
    float* kernel = kernel_;
    for (int z = 0; z < size.z; z++) {
        Int3 coord = Int3(x, y, z);
        
        fetchWindow(voxelcopy, kernel, coord, size);

        if (voxels[xyzToIndex(coord, size)].ignore)
            continue;

        float best_mean = -1;// voxels[xyzToIndex(coord, size)].norm_val;
        float lowest_var = 999;
        for (int i = 30; i < 43; i++) {
            float var = masks[i].applyMask(kernel);
            if (var < lowest_var) {
                lowest_var = var;
                best_mean = masks[i].mean;
            }
        }
        if (best_mean != -1)
            voxels[xyzToIndex(coord, size)].norm_val = best_mean;  
    }
}

void Preprocessor::rmf(Volume* vol) {
    auto start = chrono::high_resolution_clock::now();

    //float* normcopy = makeNormvalCopy(vol);
    Voxel* voxelcopy;
    cudaMallocManaged(&voxelcopy, vol->len * sizeof(Voxel));
    cudaMemcpy(voxelcopy, vol->voxels, vol->len * sizeof(Voxel), cudaMemcpyDeviceToDevice);
    printf("Copy made in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));


    CudaMask* masks = new CudaMask[43];
    CudaMask* gpu_masks;
    makeMasks(masks);
    cudaMallocManaged(&gpu_masks, 43 * sizeof(CudaMask));
    cudaMemcpy(gpu_masks, masks, 43 * sizeof(CudaMask), cudaMemcpyHostToDevice);

    rotatingMaskFilterKernel << <vol->size.y, vol->size.x >> > (vol->voxels, voxelcopy, vol->size, vol->CB->compact_gpu, gpu_masks);
    cudaDeviceSynchronize();
    
    delete(masks);
    cudaFree(voxelcopy);
    cudaFree(gpu_masks);
    printf("RMF applied in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));

}




















//------------------------------------------------------------------------------------------------------------------K MEANS --------------------------------------------------------------------------------//

__managed__ float kcluster_total_change;

//--------------------------KERNEL Helper functions----------------------//
__device__ void updateGlobalClustersIntoShared(CudaKCluster* shared_clusters, CudaKCluster* kclusters, int k) {
    int x = threadIdx.x;

    if (x < k) {
        shared_clusters[x] = kclusters[x];
    }
}

__device__ void resetAccumulations(float* thread_accs, short* thread_mems, int k) {    // accs->accumulations, mems->num. members  
    int x = threadIdx.x;    
    int offset = x * k;
    for (int i = 0; i < k; i++) {
        thread_accs[offset + i] = 0;
        thread_mems[offset + i] = 0;
    }
}

__device__ void updateSharedMemClusters(CudaKCluster* shared_clusters, float* thread_accs, short* thread_mems, int k, Int3 size) {      // Here it is vital we sync, both before and after updating!
    int x = threadIdx.x;
    __syncthreads();
    if (x < k) {
        for (int i = 0; i < size.x; i++) {
            int index = x + i * k;
            shared_clusters[x].assignBatch(thread_accs[index], thread_mems[index]);
        }
    }
    __syncthreads();
}

__device__ void pushSharedMemClusterToGlobalBlockClusters(CudaKCluster* shared_clusters, CudaKCluster* global_clusters, int k) {
    int y = blockIdx.x;
    int x = threadIdx.x;
    if (x < k)
        global_clusters[y * k + x] = shared_clusters[x];
    __syncthreads();                                                    // Only relevant if we do NOT exit kernel afterwards
}




//--------------------------Kernels----------------------//
__global__ void kMeansAssignmentKernel(Voxel* voxels, CudaKCluster* kclusters, CudaKCluster* global_clusters, int k, Int3 size) {
    int y = blockIdx.x;
    int x = threadIdx.x;

    extern __shared__ CudaKCluster block_clusters[];
    CudaKCluster* shared_clusters = (CudaKCluster*)block_clusters;
    float* thread_accs = (float*)&shared_clusters[k];
    short* thread_mems = (short*)&thread_accs[k*size.x];

    
    updateGlobalClustersIntoShared(shared_clusters, kclusters, k);  // Load clusters into shared mem  
    resetAccumulations(thread_accs, thread_mems, k);                // Init thread mem
   
    // Algo
    int offset = x * k;
    int count = 0;
    for (int z = 0; z < size.z; z++) {
        Voxel voxel = voxels[xyzToIndex(Int3(x,y,z), size)];
        if (!voxel.ignore) {
            thread_accs[offset + (count % k)] += voxel.norm_val;
            thread_mems[offset + (count % k)] += 1;
            count++;
        }
    }

    updateSharedMemClusters(shared_clusters, thread_accs, thread_mems, k, size);
    pushSharedMemClusterToGlobalBlockClusters(shared_clusters, global_clusters, k);
}




__global__ void kMeansRunKernel(Voxel* voxels, CudaKCluster* kclusters, CudaKCluster* global_clusters, int k, Int3 size) {
    int y = blockIdx.x;
    int x = threadIdx.x;

    extern __shared__ CudaKCluster block_clusters[];
    CudaKCluster* shared_clusters = (CudaKCluster*)block_clusters;
    float* thread_accs = (float*)&shared_clusters[k];
    short* thread_mems = (short*)&thread_accs[k * size.x];

    updateGlobalClustersIntoShared(shared_clusters, kclusters, k);  // Load clusters into shared mem  
    resetAccumulations(thread_accs, thread_mems, k);                // Init thread mem

    // Algo
    int offset = x * k;
    for (int z = 0; z < size.z; z++) {
        Voxel voxel = voxels[xyzToIndex(Int3(x, y, z), size)];
        if (!voxel.ignore) {
            float highest_belonging = 0;
            int best_index = 0;

            for (int i = 0; i < k; i++) {
                float belonging = shared_clusters[i].belonging(voxel.norm_val);
                if (belonging > highest_belonging) {
                    highest_belonging = belonging;
                    best_index = i;
                }
            }
            int index = offset + best_index;
            thread_accs[index] += voxel.norm_val;
            thread_mems[index] += 1;
        }
    }

    updateSharedMemClusters(shared_clusters, thread_accs, thread_mems, k, size);
    pushSharedMemClusterToGlobalBlockClusters(shared_clusters, global_clusters, k);
}

__global__ void updateGlobalClustersKernel(CudaKCluster* kclusters, CudaKCluster* block_clusters, int k, Int3 size) { // block_clusters are in global memory
    int y = blockIdx.x;
    int x = threadIdx.x;    
    kcluster_total_change = 0;
    for (int i = 0; i < size.y; i++) {
        kclusters[x].mergeBatch(block_clusters[i * k + x]);
    }
    if (y == 0 && x < k) {
        kcluster_total_change+= kclusters[x].calcCentroid();
    }
}

__global__ void testKernel() {}







void printStuff(CudaKCluster* cluster_dev, int k) {
    CudaKCluster* kc_host = new CudaKCluster[k];
    cudaMemcpy(kc_host, cluster_dev, k * sizeof(CudaKCluster), cudaMemcpyDeviceToHost);                       // Basically just copies the id
    for (int i = 0; i < k; i++) {
        CudaKCluster kc = kc_host[i];
        printf("    K-Cluster %02d	centroid: %f    members: %d\n", kc.id, kc.centroid, kc.prev_members);

    }
}



void updateKClusters(CudaKCluster* kclusters, CudaKCluster* kclusters_blocks, Int3 size, int k) {
    CudaKCluster* kcb_host = new CudaKCluster[k*size.y];
    CudaKCluster* kc_host = new CudaKCluster[k];
    cudaMemcpy(kcb_host, kclusters_blocks, size.y * k * sizeof(CudaKCluster), cudaMemcpyDeviceToHost);      // Main data transfer
    cudaMemcpy(kc_host, kclusters, k * sizeof(CudaKCluster), cudaMemcpyDeviceToHost);                       // Basically just copies the id
    for (int i = 0; i < k*size.y; i++) { 
        //printf("%02d    %02d    %f\n", kcb_host[i].id, kcb_host[i].num_members, kcb_host[i].accumulation);
        kc_host[i%k].mergeBatch(kcb_host[i]); 
    }
    for (int i = 0; i < k; i++) {
        kc_host[i].calcCentroid();
    }

    cudaMemcpy(kclusters, kc_host, k * sizeof(CudaKCluster), cudaMemcpyHostToDevice);
}

CudaKCluster* initClusters(int k) {
    CudaKCluster* kclusters_host = new CudaKCluster[k];
    for (int i = 0; i < k; i++)
        kclusters_host[i] = CudaKCluster(i);
    CudaKCluster* kclusters_device;
    cudaMallocManaged(&kclusters_device, k * sizeof(CudaKCluster));
    cudaMemcpy(kclusters_device, kclusters_host, k * sizeof(CudaKCluster), cudaMemcpyHostToDevice);
    return kclusters_device;
}



CudaKCluster* Preprocessor::kMeans(Volume* vol, int k) {                                    // We must launch separate kernels to update clusters. Only 100% safe way to sync threadblocks!
    auto start = chrono::high_resolution_clock::now();

    CudaKCluster* kclusters_device = initClusters(k);
    CudaKCluster* kclusters_blocks;
    cudaMallocManaged(&kclusters_blocks, vol->size.y * k * sizeof(CudaKCluster));   // K clusters for each BLOCK

    int shared_mem_size = k * sizeof(CudaKCluster) + k * vol->size.x * sizeof(float) +k * vol->size.x * sizeof(short);
    printf("Allocating %d Kb of mem per block\n", shared_mem_size / 1000);
    kMeansAssignmentKernel << <vol->size.y, vol->size.x, shared_mem_size >> > (vol->voxels, kclusters_device, kclusters_blocks, k, vol->size);
    cudaDeviceSynchronize();
    checkCudaError();
    
    kcluster_total_change = 99;
    while (kcluster_total_change > 0.2) {
        printf("Total change for kclusters: %f\n", kcluster_total_change);

        kMeansRunKernel << <vol->size.y, vol->size.x >> > (vol->voxels, kclusters_device, kclusters_blocks, k, vol->size);
        cudaDeviceSynchronize();
        printf("Total change for kclusters: %f\n", kcluster_total_change);

        updateGlobalClustersKernel << <1, k >> > (kclusters_device, kclusters_blocks, k, size);
        cudaDeviceSynchronize();
        printf("Total change for kclusters: %f\n", kcluster_total_change);
    }
    


    cudaDeviceSynchronize();

    printStuff(kclusters_device, k);

    //updateKClusters(kclusters_device, kclusters_blocks, vol->size, k);

    printf("KCluster found in %d ms.\n\n\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));
    return kclusters_device;
}




































// RESIZING

__device__ bool isLegal(Int3 coord, Int3 size) {
    return (coord.x >= 0 && coord.y >= 0 && coord.z >= 0 && coord.x < size.x&& coord.y < size.y&& coord.z < size.z);
}

__device__ float cubicInterpolate(float p[4], float x) {
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

__device__ float bicubicInterpolate(float p[4][4], float x, float y) {
    float arr[4];
    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}

__device__ float tricubicInterpolate(float p[4][4][4], float x, float y, float z) {
    float arr[4];
    arr[0] = bicubicInterpolate(p[0], y, z);
    arr[1] = bicubicInterpolate(p[1], y, z);
    arr[2] = bicubicInterpolate(p[2], y, z);
    arr[3] = bicubicInterpolate(p[3], y, z);
    return cubicInterpolate(arr, x);
}

__device__ void moveDown(float* kernel) {
    for (int i = 0; i < 4 * 4 * 3; i++) {
        kernel[i] = kernel[i + 4 * 4];
    }
}
__device__ void getInterpolationKernel(float* raw_scans, float* kernel, int x, int y, int z, Int3 size) {
    int i = 0;
    int z_start = z;

    if (z > 0) {               // WAAAAY fewer array acesses!
        moveDown(kernel);
        z_start = z + 3;
        i = 4 * 4 * 3;
    }

    for (int z_ = z_start; z_ < z + 4; z_++) {
        for (int y_ = y; y_ < y + 4; y_++) {
            for (int x_ = x; x_ < x + 4; x_++) {
                if (isLegal(Int3(x_, y_, z_), size)) {
                    float a = raw_scans[xyzToIndex2(Int3(x_, y_, z_), size)];
                    kernel[i] = a;
                }
                else
                    kernel[i] = -1000;
                i++;
            }
        }
    }
}

__global__ void interpolationKernel(float* raw_scans, float* resized_scan, Int3 size_from, Int3 size_to, float z_over_xy) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    float kernel[4 * 4 * 4];
    float kernel_arr[4][4][4];

    int z_new = 0;
    for (int z = -1; z <= size_from.z; z++) {


        getInterpolationKernel(raw_scans, kernel, x, y, z, size_from);
        for (int i = 0; i < 64; i++) {
            kernel_arr[i / 16][(i % 16) / 4][i % 4] = kernel[i];
        }

        while (z_new / z_over_xy < (float)(z + 2)) {
            float z_new_mapped = z_new / z_over_xy;
            float rel_z = (z_new_mapped - z) / 3.;	//Image on phone for explanation

            // No xy increase
            Int3 coord(x, y, z_new);
            if (isLegal(coord, size_to)) {
                float point_val = tricubicInterpolate(kernel_arr, 3 / 6., 3 / 6., rel_z);
                int point_index = xyzToIndex2(coord, size_to);
                resized_scan[point_index] = point_val;
            }
            /*for (int yoff = 0; yoff < 2; yoff++) {
                for (int xoff = 0; xoff < 2; xoff++) {
                    Int3 coord(x * 2 + xoff, y * 2 + yoff, z_new);
                    if (isLegal(coord, size_to)) {
                        float point_val = tricubicInterpolate(kernel_arr, 2 / 6. + xoff / 6., 2 / 6. + yoff / 6., rel_z);
                        int point_index = xyzToIndex2(coord, size_to);
                        resized_scan[point_index] = point_val;
                    }
                }
            }*/
            z_new++;
        }
    }
}


float* Interpolate3D(float* raw_scans, Int3 size_from, Int3* size_out, float z_over_xy) {
    auto start = chrono::high_resolution_clock::now();



    int num_slices_ = (int)(size_from.z * z_over_xy);
    Int3 size_to(512, 512, num_slices_);
    int len1D_new = size_to.x * size_to.y * size_to.z;
    int len1D_old = size_from.x * size_from.y * size_from.z;

    float* resized_scan = new float[len1D_new]();    // Reserve space on Host side

    printf("Resizing from:		%d %d %d   to %d %d %d\n", size_from.x, size_from.y, size_from.z, size_to.x, size_to.y, size_to.z);
    printf("Required VRAM: %d Mb\n", (len1D_old * sizeof(float) + len1D_new * sizeof(float)) / 1000000);

    float* raws;
    float* resiz;
    cudaMallocManaged(&raws, len1D_old * sizeof(float));
    cudaMallocManaged(&resiz, len1D_new * sizeof(float));

    cudaMemcpy(raws, raw_scans, len1D_old * sizeof(float), cudaMemcpyHostToDevice);
    interpolationKernel << <size_to.y, size_to.x >> > (raws, resiz, size_from, size_to, z_over_xy);   
    cudaError_t err = cudaGetLastError();        // Get error code
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    cudaMemcpy(resized_scan, resiz, len1D_new * sizeof(float), cudaMemcpyDeviceToHost);

    // Free up memory
    cudaFree(raws);
    cudaFree(resiz);
    delete(raw_scans);

    printf("Resizing completed in %d ms.\n\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));

    // Returning values
    *size_out = size_to;
    return resized_scan;
}