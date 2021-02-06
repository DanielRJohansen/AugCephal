#include "FuzzyAssignment.cuh"

//------------------------------------------------------------------------------------------------------------------K MEANS --------------------------------------------------------------------------------//

__managed__ float kcluster_total_change = 99;

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
__device__ void resetBelongings(float* belongings, int k) {
    for (int i = 0; i < k; i++) {
        belongings[i] = 0;
    }
}

__device__ int getBestBelongingIndex(float* belongings, int k) {
    int best_index = 0;
    float best_belonging = 0;
    for (int i = 0; i < k; i++) {
        if (belongings[i] > best_belonging) {
            best_belonging = belongings[i];
            best_index = i;
        }
    }
    return best_index;
}


//--------------------------Kernels----------------------//
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
    int x = threadIdx.x;

    extern __shared__ float change_arr[];
    float* shared_change = (float*)change_arr;

    for (int i = 0; i < size.y; i++) {
        kclusters[x].mergeBatch(block_clusters[i * k + x]);
    }

    shared_change[x] = kclusters[x].calcCentroid();;

    __syncthreads();

    if (x == 0) {
        kcluster_total_change = 0;
        for (int i = 0; i < k; i++)
            kcluster_total_change += shared_change[i];
    }
}

__global__ void fuzzyAssignmentKernel(Voxel* voxels, CudaKCluster* kclusters, float* gauss_kernel, int k, Int3 size) {
    int y = blockIdx.x;
    int x = threadIdx.x;

    int thread_offset = x * k;
    extern __shared__ float block_belongings[];
    float* belongings = (float*)&block_belongings[thread_offset];


    for (int z = 0; z < size.z; z++) {
        resetBelongings(belongings, k);
        int neighbor_ignores = 0;
        Voxel voxel = voxels[xyzToIndex(Int3(x, y, z), size)];
        if (!voxel.ignore) {
            for (int z_ = z - 1; z_ <= z + 1; z_++) {
                for (int y_ = y - 1; y_ <= y + 1; y_++) {
                    for (int x_ = x - 1; x_ <= x + 1; x_++) {
                        if (isInVolume(Int3(x_, y_, z_), size)) {
                            int gauss_kernel_index = (z_ - z) * 9 + (y_ - y) * 3 + (x_ - x);
                            Voxel voxel_ = voxels[xyzToIndex(Int3(x_, y_, z_), size)];
                            if (!voxel_.ignore) {
                                for (int i = 0; i < k; i++) {
                                    belongings[i] += kclusters[i].belonging(voxel_.norm_val) * gauss_kernel[gauss_kernel_index];
                                }
                            }
                            else
                                neighbor_ignores++;
                        }
                    }
                }
            }


            int best_index = getBestBelongingIndex(belongings, k);
            voxel.color = kclusters[best_index].color;
            //voxel.norm_val = kclusters[best_index].centroid;
            voxel.kcluster = best_index;
            if (neighbor_ignores > 7)
                voxel.ignore = true;
            voxels[xyzToIndex(Int3(x, y, z), size)] = voxel;
        }
    }
}















//------------------------------------------------------HOST Helper functions--------------------------------------------------------------------------------------//
void checkCudaError() {
    cudaError_t err = cudaGetLastError();        // Get error code
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


int inline normvalToHuval(float norm) {return norm * 1500. - 700;}

void printKmeansStuff(CudaKCluster* cluster_dev, int k) {
    CudaKCluster* kc_host = new CudaKCluster[k];
    cudaMemcpy(kc_host, cluster_dev, k * sizeof(CudaKCluster), cudaMemcpyDeviceToHost);                       // Basically just copies the id
    printf("\n");
    for (int i = 0; i < k; i++) {
        CudaKCluster kc = kc_host[i];
        printf("    K-Cluster %02d	centroid: %05d    members: %d       color: %03d %03d %03d\n", kc.id, normvalToHuval(kc.centroid), kc.prev_members), kc.color.r, kc.color.g, kc.color.b;
    }
    printf("\n");
}


CudaKCluster* initClusters(int k) {
    CudaKCluster* kclusters_host = new CudaKCluster[k];
    for (int i = 0; i < k; i++)
        kclusters_host[i] = CudaKCluster(i, k);
    CudaKCluster* kclusters_device;
    cudaMallocManaged(&kclusters_device, k * sizeof(CudaKCluster));
    cudaMemcpy(kclusters_device, kclusters_host, k * sizeof(CudaKCluster), cudaMemcpyHostToDevice);
    return kclusters_device;
}




float dist(Int3 o, Int3 p) {
    float x_ = o.x - p.x;
    float y_ = o.y - p.y;
    float z_ = o.z - p.z;
    return sqrt(x_ * x_ + y_ * y_ + z_ * z_);
}

float* makeGaussianKernel3D() {
    float* kernel = new float[3 * 3 * 3];
    Int3 o(0, 0, 0);
    int index = 0;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Int3 p(x, y, z);
                kernel[index++] = 1 / (1 + dist(o, p));
            }
        }
    }
    float* kernel_dev;
    cudaMallocManaged(&kernel_dev, 3 * 3 * 3 * sizeof(float));
    cudaMemcpy(kernel_dev, kernel, 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    delete(kernel);
    return kernel_dev;
}



void checkFuzzyAssignment(Volume* vol, int k) {
    Voxel* vh = new Voxel[vol->len];
    cudaMemcpy(vh, vol->voxels, vol->len * sizeof(Voxel), cudaMemcpyDeviceToHost);
    int* belongings = new int[k+1]();
    for (int i = 0; i < vol->len; i++) {
        belongings[vh[i].kcluster+1] += 1;
    }
    printf("Fuzzy assignment: \n");
    for (int i = -1; i < k; i++)
        printf("Kluster %d  members: %d\n", i, belongings[i+1]);

}





//---------------------------------------------KERNEL launchers -----------------------------------------------------------------------------------------------------------------//



CudaKCluster* FuzzyAssigner::kMeans(Volume* vol, int k, int max_iterations) {                                    // We must launch separate kernels to update clusters. Only 100% safe way to sync threadblocks!
    auto start = chrono::high_resolution_clock::now();

    CudaKCluster* kclusters_device = initClusters(k);
    CudaKCluster* kclusters_blocks;
    cudaMallocManaged(&kclusters_blocks, vol->size.y * k * sizeof(CudaKCluster));   // K clusters for each BLOCK

    int shared_mem_size = k * sizeof(CudaKCluster) + k * vol->size.x * sizeof(float) + k * vol->size.x * sizeof(short);
    printf("\n\nExecuting kMeans with %d clusters.\nAllocating %d Kb of memory per threadblock\n", k, shared_mem_size / 1000);



    int iterations = 0;
    while (kcluster_total_change > 0.002 && iterations < max_iterations) {
        kMeansRunKernel << <vol->size.y, vol->size.x, shared_mem_size >> > (vol->voxels, kclusters_device, kclusters_blocks, k, vol->size);
        checkCudaError();
        cudaDeviceSynchronize();

        updateGlobalClustersKernel << <1, k, k * sizeof(float) >> > (kclusters_device, kclusters_blocks, k, vol->size);
        cudaDeviceSynchronize();

        printf("Total change for kclusters: %f    iterations: %02d\r", kcluster_total_change, iterations++);
    }
    printKmeansStuff(kclusters_device, k);


    printf("KCluster found in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));
    return kclusters_device;
}


void FuzzyAssigner::fuzzyClusterAssignment(Volume* vol, CudaKCluster* kclusters_dev, int k) {
    auto start = chrono::high_resolution_clock::now();


    float* gauss_kernel_dev = makeGaussianKernel3D();
    int shared_mem_size = k * vol->size.x * sizeof(float);
    fuzzyAssignmentKernel << <vol->size.y, vol->size.x, shared_mem_size >> > (vol->voxels, kclusters_dev, gauss_kernel_dev, k, vol->size);
    cudaDeviceSynchronize();

    //checkFuzzyAssignment(vol, k);

    printf("Fuzzy assignment completed in %d ms.\n\n\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));

}