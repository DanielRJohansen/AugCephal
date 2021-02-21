#include "Rendering.cuh"

void checkCudaError2() {
    cudaError_t err = cudaGetLastError();        // Get error code
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

Ray* RenderEngine::initRays() {
    Ray* rayptr_host = new Ray[NUM_RAYS];
    float rpd = (float)RAYS_PER_DIM;
    for (int y = 0; y < RAYS_PER_DIM; y++) {
        for (int x = 0; x < RAYS_PER_DIM; x++) {
            float x_ = 0.5 - 0.5 / rpd - x / rpd;// Shift by half increment to have
            float y_ = 0.5 - 0.5 / rpd - y / rpd;
            float d = sqrt(FOCAL_LEN * FOCAL_LEN + x_ * x_ + y_ * y_);
            rayptr_host[xyToRayIndex(y, x)] = Ray(Float3(x_, y_, FOCAL_LEN) * (1. / d));	// Yes xy is swapped, this works, so schhh!
        }
    }
    return rayptr_host;
};





__device__ float activationFunction(float counts) {
    return 2 / (1 + powf(e, (-counts / 4.))) - 1.;
}

/*__device__ float lightSeeker(Block* volume, CudaFloat3 pos) {
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
                if (isInVolume(Int3(vol_x, vol_y, vol_z))) {
                    int index = xyzToIndex(vol_x, vol_y, vol_z);
                    if (!volume[index].ignore) { brightness -= 1; break; }
                    //else { break; }
                }
                //else { brightness += 1; }
            }
        }
    }
    return activationFunction(brightness);
}*/

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

    CudaFloat3 vector = CudaFloat3(x_z, y_z, z_y);
    vector.norm();
    return vector;
}


__device__ bool isInVolumeCF3(CudaFloat3 relativeorigin, Int3 vol_size) {
    float buf = 0.001;
    return relativeorigin.x + buf >= 0 && relativeorigin.y + buf >= 0 && relativeorigin.z + buf >= 0 && relativeorigin.x - buf <= vol_size.x && relativeorigin.y - buf <= vol_size.y && relativeorigin.z - buf <= vol_size.z;
}

__device__ float overwriteIfLowerAndIntersect(Int3 size, CudaFloat3 origin, CudaFloat3 vector, float old, float newval, float above) {
    if (newval < old && newval > above) {
        if (isInVolumeCF3(origin+(vector*newval), size))
            return newval;
    }
    return old;
}
__device__ float calcDist(float origin, float vec, float point) {
    if (abs(vec) < 0.0001)
        return 9999;
    float dist = (point - origin) / vec;
    if (dist < 0)
        dist = 0;
    return dist;
}


__device__ float firstIntersection(CudaFloat3 origin, CudaFloat3 vector, Int3 size, float above) {
    float lowest = 4000;
    lowest = overwriteIfLowerAndIntersect(size, origin, vector, lowest, calcDist(origin.x, vector.x, size.x), above);
    lowest = overwriteIfLowerAndIntersect(size, origin, vector, lowest, calcDist(origin.y, vector.y, size.y), above);
    lowest = overwriteIfLowerAndIntersect(size, origin, vector, lowest, calcDist(origin.z, vector.z, size.z), above);
    lowest = overwriteIfLowerAndIntersect(size, origin, vector, lowest, calcDist(origin.x, vector.x, 0), above);
    lowest = overwriteIfLowerAndIntersect(size, origin, vector, lowest, calcDist(origin.y, vector.y, 0), above);
    lowest = overwriteIfLowerAndIntersect(size, origin, vector, lowest, calcDist(origin.z, vector.z, 0), above);

    return lowest;
}

__device__ Int2 smartStartStop(CudaFloat3 origin, CudaFloat3 vector, Int3 vol_size) {
    CudaFloat3 size = CudaFloat3(vol_size);
    CudaFloat3 rel_origin = origin + (size * 0.5);

    float start, stop;
    if (isInVolumeCF3(rel_origin, vol_size))
        start = 50;
    else
        start = firstIntersection(rel_origin, vector, vol_size, 0);
    stop = firstIntersection(rel_origin, vector, vol_size, start);
    return Int2(start, stop);
}



//--------------------------------------------------------------------------------------    KERNEL  --------------------------------------------------------------------------------------------------------------------------//

//texture<CompactCluster, 1, cudaReadModeElementType> clusters_texture;

__global__ void stepKernel(Ray* rayptr, CompactCam cc, int offset, uint8_t* image, Int3 vol_size, unsigned* ignores, int ignores_len, RenderVoxel* rendervoxels, CompactCluster* compactclusters, int num_clusters) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x +offset;
    //50
    Ray ray = rayptr[index];    // This operation alone takes ~60 ms


    extern __shared__ unsigned int blockignores[];
    unsigned int* xyignores = (unsigned int*)blockignores;
    int temp = threadIdx.x;
    while (temp < ignores_len) {
        xyignores[temp] = ignores[temp];
        temp += THREADS_PER_BLOCK;
    }
    
    __syncthreads();



    CudaFloat3 unit_vector = makeUnitVector(&ray, cc);
    CudaRay cray(unit_vector * RAY_SS);
    CompactBool CB;


    //77
    Int2 start_stop = smartStartStop(CudaFloat3(cc.origin.x, cc.origin.y, cc.origin.z), cray.step_vector, vol_size);
    int prev_vol_index = -1;
    int prev_cluster_id = -1;
    for (int step = start_stop.x; step < start_stop.y+1; step++) {    //500
        int x = cc.origin.x + cray.step_vector.x * step;
        int y = cc.origin.y + cray.step_vector.y * step;
        int z = cc.origin.z + cray.step_vector.z * step;

        int vol_x = (int) (x + vol_size.x / 2);
        int vol_y = (int) (y + vol_size.y / 2);
        int vol_z = (int) (z + vol_size.z / 2);
        Int3 pos = Int3(vol_x, vol_y, vol_z);
        

        if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && vol_x < vol_size.x && vol_y < vol_size.y && vol_z < vol_size.z) { 
            int volume_index = xyzToIndex(pos, vol_size);            
            if (volume_index == prev_vol_index)
                continue;

            int column_index = vol_y * vol_size.x + vol_x;
            if (CB.getBit(xyignores, column_index) != 0)
                continue;
            
            int cluster_id = rendervoxels[volume_index].cluster_id;
            if (cluster_id == prev_cluster_id) {
                continue;
            }
            prev_cluster_id = cluster_id;

            if (cluster_id == -1)
                continue;
            CompactCluster* compactcluster = &compactclusters[cluster_id];
            cray.color.add(compactcluster->getColor() * compactcluster->getAlpha());
            cray.alpha += compactcluster->getAlpha();
            if (cray.alpha >= 1)
                break;
        }
    }
    cray.color.cap();   //Caps each channel at 255

    image[index * 4 + 0] = (int)cray.color.r;
    image[index * 4 + 1] = (int)cray.color.g;
    image[index * 4 + 2] = (int)cray.color.b;
    image[index * 4 + 3] = 255;
}









//--------------------------------------------------------------------------------------    LAUNCHER     --------------------------------------------------------------------------------------------------------------------//


void RenderEngine::render(sf::Texture* texture) {
    auto start = chrono::high_resolution_clock::now();
    
    CompactCam cc = CompactCam(camera->origin, camera->plane_pitch, camera->plane_yaw, camera->radius);


    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&(stream[i]));
    }


    for (int i = 0; i < N_STREAMS; i++) {       // Needed because we dont want to block the GPU from other requests!
        int offset = i * stream_size;
        cudaMemcpyAsync(&rayptr_device[offset], &rayptr_host[offset], ray_stream_bytes, cudaMemcpyHostToDevice, stream[i]);
    }
    /*int* f_device;
    unsigned icopy[8192];
    for (int i = 0; i < 8192; i++)
        icopy[i] = hostignores[i];
    cudaMallocManaged(&f_device, sizeof(int));*/

    int shared_mem_size = volume->CB->arr_len*sizeof(unsigned int);
    printf("memsize: %d\n", shared_mem_size);
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        stepKernel << <blocks_per_sm, THREADS_PER_BLOCK, shared_mem_size, stream[i] >> > (rayptr_device, cc, offset, image_device, volume->size, compactignores, volume->CB->arr_len, volume->rendervoxels, volume->compactclusters, volume->num_clusters);
        checkCudaError2();
    }

    printf("Rendering...");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&image_host[offset * 4], &image_device[offset * 4], image_stream_bytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    

    cudaDeviceSynchronize();
    texture->update(image_host);

    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    printf("Executed in %d ms.\n", duration);
}


