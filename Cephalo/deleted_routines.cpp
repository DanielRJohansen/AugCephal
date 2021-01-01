/*
* 
* 
*
* 
* 
*


void VolumeMaker::setIgnores(vector<int> ignores) {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        for (int j = 0; j < 6; j++) {
            if (ignores[volume[i].cat_index]) {
                volume[i].ignore = true;
                break;
            }
        }
    }
}















void SliceMagic::mergeClusters(Pixel* image, int num_clusters, float max_absolute_dist, float max_fractional_dist) {
    printf("Merging clusters. RAM req: %d Kb\n", num_clusters * sizeof(TissueCluster) / 1000);
    //TissueCluster* TC = initTissueCluster(image, num_clusters, sizesq);
    TissueCluster* TC = new TissueCluster[num_clusters]();

    for (int i = 0; i < sizesq; i++) {
        if (!image[i].isEdge())
            continue;

        int* connected_indexes = new int[9];
        int num_connected = image[i].connectedClusters(image, connected_indexes);

        if (num_connected == 0)
            continue;

        int survivor_index = connected_indexes[0];
        TissueCluster survivor_copy = TC[survivor_index];

        if (num_connected == 1) {
            image[i].assignToCluster(survivor_copy.cluster_id, survivor_copy.color, survivor_copy.cluster_mean);
            TC[connected_indexes[0]].addToCluster(image[i]);
        }
        else {
            // The first cluster (at index 0) will be only surviving cluster, others deathmarked. All pixels will belong to first cluster;
            TissueCluster** sublist = new TissueCluster* [num_connected];
            makeClusterSublist(TC, sublist, connected_indexes, num_connected);

            bool mergeable = TC[survivor_index].isMergeable(sublist, num_connected, max_absolute_dist, max_fractional_dist);
            if (mergeable) {
                TC[survivor_index].mergeClusters(sublist, image, num_connected);
                TC[survivor_index].addToCluster(image[i]);
                image[i].assignToCluster(TC[survivor_index].cluster_id, TC[survivor_index].color, TC[survivor_index].cluster_mean);
            }
            delete(sublist);    // Not properly deleted!!!!
        }
        //else
            //image[i].color = Color3(0, 0, 0);
        delete(connected_indexes);
    }
    delete(TC);
}



TissueCluster::TissueCluster(Pixel p) {
    min_val = p.getClusterMean();
    max_val = p.getClusterMean();
    //printf("Initting cluster from mean: %f\n", p.cluster_mean);
    cluster_mean = p.getClusterMean();
    color = p.color;
    initialized = true;
    cluster_id = p.cluster_id;
    //printf("Initializing cluster %d\n", cluster_id);
    addToCluster(p);
}























void VolumeMaker::categorizeBlocks() {
    for (int z = 0; z < VOL_Z; z++) {
        //printf(" \r Categorizing z level %d ", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (volume[block_index].ignore) {
                    volume[block_index].cat_index = colorscheme.cat_indexes[0];
                    volume[block_index].prev_cat_index = volume[block_index].cat_index;
                }
                else {
                    int hu_index = (int)((volume[block_index].value * (HU_MAX - HU_MIN - 1)));
                    //if (hu_index > 200)
                     //   printf("%f      %d \n",  volume[block_index].value, hu_index);
                    volume[block_index].cat_index = colorscheme.cat_indexes[hu_index];
                    volume[block_index].prev_cat_index = volume[block_index].cat_index;
                }

            }
        }
    }
}




















void CudaOperator::rayStepMS(Ray* rp, CompactCam cc) {
    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&(stream[i]));
    }
    printf("Sending\n");


    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&rayptr[offset], &rp[offset], stream_bytes, cudaMemcpyHostToDevice, stream[i]);
    }


    printf("to device\n");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        stepKernelMS << <blocks_per_sm, THREADS_PER_BLOCK, 0, stream[i] >> > (rayptr,
            blocks, cc, offset);
    }
    printf("execution\n");
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * stream_size;
        cudaMemcpyAsync(&rp[offset], &rayptr[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[i]);

    }

    printf("Received\n");
    cudaDeviceSynchronize();
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }
}

















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
void CudaOperator::rayStep(Ray *rp) {
    // Copy rayptr to device
    cudaMemcpy(rayptr, rp, NUM_RAYS * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    stepKernel << <1024, 1024 >> > (rayptr, blocks, &ray_block[0]);    // RPD blocks (y), RPD threads(x)

    cudaDeviceSynchronize();

    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(rp, rayptr, NUM_RAYS * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}





void VolumeMaker::medianFilter() {
    Block* vol_copy = copyVolume(volume);
    for (int z = 0; z < VOL_Z; z++) {
        printf("Filtering layer %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {

                int block_index = xyzToIndex(x, y, z);
                if (x * y * z == 0 || x == VOL_X - 1 || y == VOL_Y - 1 || z == VOL_Z - 1) {    // Set all edges to air to no (out of mem problems)
                    volume[block_index].air = true;
                }
                else {
                    //float window_values[27];
                    vector <float>window(27);
                    int i = 0;
                    for (int z_off = -1; z_off < 2; z_off++) {
                        for (int y_off = -1; y_off < 2; y_off++) {
                            for (int x_off = -1; x_off < 2; x_off++) {
                                window[i] = vol_copy[xyzToIndex(x + x_off, y + y_off, z + z_off)].value;
                                //cout << window[i] << endl;
                                i++;
                            }
                        }
                    }
                    sort(window.begin(), window.end());
                    volume[block_index].value = window[14];
                }
            }
        }
    }
}








*/