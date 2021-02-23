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





/*__device__ float activationFunction(int counts) {
    return 1. / (1 + powf(e, (-counts / 4.))) - 1.;
}*/
__device__ float activationFunction(int counts) {
    return 1./0.6 * (    1. / (1 + exp(-counts / 2.))    -0.4);
}
__device__ float lightSeeker(short int cluster_id, RenderVoxel* voxels, Int3 vol_pos, Int3 vol_size) {
    int clear_voxels = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            Int3 pos_ = vol_pos + Int3(x, y, 1);              
            if (isInVolume(pos_, vol_size)) {
                int voxel_index = xyzToIndex(pos_, vol_size);
                if (voxels[voxel_index].cluster_id != cluster_id) { 
                    clear_voxels++;
                }
            }
            else {
                clear_voxels++;
            }
            
        }
    }
    return activationFunction(clear_voxels);
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


__global__ void stepKernel(Ray* rayptr, CompactCam cc, uint8_t* image, Int3 vol_size, unsigned* ignores, int ignores_len, RenderVoxel* rendervoxels, CompactCluster* compactclusters, int num_clusters) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
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



    short int clusterids_hit[5];
    for (int i = 0; i < 5; i++)
        clusterids_hit[i] = -1;
    int hit_index = 0;

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
            
            short int cluster_id = rendervoxels[volume_index].cluster_id;
            if (cluster_id == prev_cluster_id) {
                continue;
            }
            prev_cluster_id = cluster_id;
           

            if (cluster_id == -1)
                continue;

            CompactCluster* compactcluster = &compactclusters[cluster_id];
            if (compactcluster->getAlpha() == 0.)
                continue;

            if (hit_index < 5) {
                clusterids_hit[hit_index] = cluster_id;
                hit_index++;
            }
            float brightness = lightSeeker(cluster_id, rendervoxels, pos, vol_size);
            cray.color.add(compactcluster->getColor() * compactcluster->getAlpha() * brightness);
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

    for (int i = 0; i < 5; i++)
        rayptr[index].clusterids_hit[i] = clusterids_hit[i];
}









//--------------------------------------------------------------------------------------    LAUNCHER     --------------------------------------------------------------------------------------------------------------------//


Ray* RenderEngine::render(sf::Texture* texture) {
    //auto start = chrono::high_resolution_clock::now();
    
    CompactCam cc = CompactCam(camera->origin, camera->plane_pitch, camera->plane_yaw, camera->radius);


    int shared_mem_size = volume->CB->arr_len * sizeof(unsigned int);
    stepKernel << <blocks_per_sm, THREADS_PER_BLOCK, shared_mem_size >> > (rayptr_device, cc, image_device, volume->size, volume->CB->compact_gpu, volume->CB->arr_len, volume->rendervoxels, volume->compactclusters, volume->num_clusters);
    cudaDeviceSynchronize();


    

    cudaMemcpyAsync(image_host, image_device, image_stream_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    texture->update(image_host);

    //printf("Executed in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));

    return rayptr_device;
}











//--------------------------------------------------------------------------------------    CONSTRUCTOR     --------------------------------------------------------------------------------------------------------------------//

texture<CompactCluster, 1, cudaReadModeElementType> clusters_texture;


RenderEngine::RenderEngine(Volume* vol, Camera* c) {

    volume = vol;
    camera = c;

    CUDAPlanning();

    //cudaMallocManaged(&voxels, vol->len * sizeof(Voxel));
    voxels = vol->voxels;
    xyColumnIgnores = vol->xyColumnIgnores;
    CB = vol->CB;
    compactignores = CB->compact_gpu;
    hostignores = CB->compact_host;
    //updateVolume();

    rayptr_host = initRays();
    cudaMallocManaged(&rayptr_device, NUM_RAYS * sizeof(Ray));
    cudaMemcpy(rayptr_device, rayptr_host, NUM_RAYS * sizeof(Ray), cudaMemcpyHostToDevice);

    cudaMallocManaged(&image_device, NUM_RAYS * 4 * sizeof(uint8_t));	//4 = RGBA
    image_host = new uint8_t[NUM_RAYS * 4];
    printf("RenderEngine initialized. Approx GPU size: %d Mb\n\n", (int)(NUM_RAYS * sizeof(Ray) / 1000000.));



    /*
    CompactCluster* compactcluster_texture;
    int bytesize = volume->num_clusters * sizeof(CompactCluster);
    cudaMalloc((void**) &compactcluster_texture, bytesize);
    cudaMemcpy(compactcluster_texture, volume->compactclusters, bytesize, cudaMemcpyDeviceToDevice);

    cudaChannelFormatDesc* channelDesc = &cudaCreateChannelDesc<int>();
    //cudaBindTexture(NULL, clusters_texture, compactcluster_texture, channelDesc, bytesize);
    size_t offset = size_t(0);
    //cudaBindTexture(&offset, clusters_texture, compactcluster_texture, channelDesc, bytesize);
    */

}