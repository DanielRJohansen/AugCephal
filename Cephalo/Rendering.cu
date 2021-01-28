#include "Rendering.cuh"



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

__device__ __host__ int xyzToIndex(Int3 coord, Int3 size) {
    return coord.z * size.y * size.x + coord.y * size.x + coord.x;
}
__device__ inline bool isInVolume(Int3 coord, Int3 size) {
    return coord.x >= 0 && coord.y >= 0 && coord.z >= 0 && coord.x < size.x&& coord.y < size.y&& coord.z < size.z;
}


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

    return CudaFloat3(x_z, y_z, z_y);
}

__global__ void testKernel(int* f) {
    *f = 99;
    return;
}


__global__ void stepKernel(Ray* rayptr, Voxel* voxels, CompactCam cc, int offset, uint8_t* image, Int3 vol_size, int* finished, bool* xyIgnores, CompactBool* CB_global, unsigned* ignores) {
    int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x +offset;
    *finished = 42;
    Ray ray = rayptr[index];    // This operation alone takes ~60 ms

    //CudaFloat3 unit_vector(x_z, y_z, z_y);
    CudaFloat3 unit_vector = makeUnitVector(&ray, cc);
    CudaRay cray(unit_vector * RAY_SS);
    CompactBool CB;

    Voxel* cached_voxel;
    cached_voxel = &voxels[0];  // Init Block, doesn't matter is never used before another is loaded.
    int prev_vol_index = -1;    // Impossible index


    for (int step = 100; step < RAY_STEPS; step++) {    //500

        int x = cc.origin.x + cray.step_vector.x * step;
        int y = cc.origin.y + cray.step_vector.y * step;
        int z = cc.origin.z + cray.step_vector.z * step;

        int vol_x = (int) (x + vol_size.x / 2);
        int vol_y = (int) (y + vol_size.y / 2);
        int vol_z = (int) (z + vol_size.z / 2);

        // Check if entire column is air, if then skip
        
          //  continue;
        //bool a = isInVolume(Int3(vol_x, vol_y, vol_z), vol_size);
        //Int3 b = Int3(vol_x, vol_y, vol_z);
        if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && vol_x < vol_size.x && vol_y < vol_size.y && vol_z < vol_size.z) { // Only proceed if coordinate is within volume!
            int volume_index = xyzToIndex(Int3(vol_x, vol_y, vol_z), vol_size);

            int column_index = vol_y * vol_size.x + vol_x;
            if (CB.getBit(ignores, column_index) == 1)
                continue;

            //if (xyIgnores[vol_y * vol_size.x + vol_x])
            //    continue;

            if (vol_z == 0) {
                float remaining_alpha = 1 - cray.alpha;
                //cray.color.r = 0;
                cray.color.g += 114 * remaining_alpha;
                cray.color.b += 158 * remaining_alpha;
                break;
            }
            //if (empty_y_slices[vol_y] || empty_x_slices[vol_x]) { continue; }

            if (volume_index == prev_vol_index) {
                if (cached_voxel->ignore) { continue; }
            }
            else {
                prev_vol_index = volume_index;
                if (voxels[volume_index].ignore) { continue; }
                else { cached_voxel = &voxels[volume_index]; }
            }


            CudaColor block_color = CudaColor(cached_voxel->color.r, cached_voxel->color.g, cached_voxel->color.b);
            float brightness = 1;// lightSeeker(voxels, CudaFloat3(vol_x, vol_y, vol_z));
            block_color = block_color * brightness;
            cray.color.add(block_color * cached_voxel->alpha);
            cray.alpha += cached_voxel->alpha;
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
    int* f_device;
    cudaMallocManaged(&f_device, sizeof(int));
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        stepKernel << <blocks_per_sm, THREADS_PER_BLOCK, 0, stream[i] >> > (rayptr_device, voxels, cc, offset, image_device, volume->size, f_device, xyColumnIgnores, CB, compactignores);// , dev_empty_y_slices, dev_empty_x_slices);
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


