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



void stepKernel(Ray* rayptr, Voxel* voxels, CompactCam* CC, int offset, uint8_t* image) {

}


void RenderEngine::render(sf::Texture* texture) {
    CompactCam CC = CompactCam(camera->origin, camera->plane_pitch, camera->plane_yaw, camera->radius);


    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&(stream[i]));
    }


    for (int i = 0; i < N_STREAMS; i++) {       // Needed because we dont want to block the GPU from other requests!
        int offset = i * stream_size;
        cudaMemcpyAsync(&rayptr_device[offset], &rayptr_host[offset], ray_stream_bytes, cudaMemcpyHostToDevice, stream[i]);
    }


    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        stepKernel << <blocks_per_sm, THREADS_PER_BLOCK, 0, stream[i] >> > (rayptr, voxels, cc, offset, image_device);// , dev_empty_y_slices, dev_empty_x_slices);
    }


    printf("Rendering...");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&image_host[offset * 4], &image_device[offset * 4], image_stream_bytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    printf("  Received!\n");
    //cudaDeviceSynchronize();
    texture->update(image_host);

    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }
}


