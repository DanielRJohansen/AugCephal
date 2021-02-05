#include "Preprocessing.cuh"

/*void checkCudaError() {
    cudaError_t err = cudaGetLastError();        // Get error code
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}*/








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









