#include "CudaOps.cuh"



__device__ int xyzToIndexI3D(Int3 coord, Int3 size) {
    return coord.z * size.y * size.x + coord.y * size.x + coord.x;
}

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
__device__ void getInterpolationKernel(float* raw_scans, float* kernel, int x, int y, int z, Int3 size) {
    int i = 0;
    for (int z_ = z; z_ < z + 4; z_++) {
        for (int y_ = y; y_ < y + 4; y_++) {
            for (int x_ = x; x_ < x + 4; x_++) {
                if (isLegal(Int3(x_, y_, z_), size)) {
                    float a = raw_scans[xyzToIndexI3D(Int3(x_, y_, z_), size)];
                    kernel[i] = a;
                }
                else
                    kernel[i] = 0;
                i++;
            }
        }
    }
}

__global__ void interpolationKernel(float* raw_scans, float* resized_scan, Int3 size_from, Int3 size_to, float z_over_xy) {
    int x = blockIdx.x;
    int y = threadIdx.x;

    float* kernel = new float[4 * 4 * 4];
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


            for (int yoff = 0; yoff < 2; yoff++) {
                for (int xoff = 0; xoff < 2; xoff++) {
                    Int3 coord(x * 2 + xoff, y * 2 + yoff, z_new);
                    if (isLegal(coord, size_to)) {
                        float point_val = tricubicInterpolate(kernel_arr, 2 / 6. + xoff / 6., 2 / 6. + yoff / 6., rel_z);
                        int point_index = xyzToIndexI3D(coord, size_to);
                        resized_scan[point_index] = point_val;
                    }
                }
            }
            z_new++;
        }
    }
    delete(kernel);
}




float* Interpolate3D(float* raw_scans, Int3 size_from, Int3* size_out, float z_over_xy) {
    auto start = chrono::high_resolution_clock::now();
    
    int num_slices_ = (int)(size_from.z * z_over_xy);
    Int3 size_to(1024, 1024, num_slices_);
    int len1D_new = size_to.x * size_to.y * size_to.z;
    int len1D_old = size_from.x * size_from.y * size_from.z;

    float* resized_scan = new float[len1D_new]();    // Reserve space on Host side

    printf("Resizing from:		%d %d %d   to %d %d %d\n", size_from.x, size_from.y, size_from.z, size_to.x, size_to.y, size_to.z);


    float* raws;
    float* resiz;
    cudaMallocManaged(&raws, len1D_old * sizeof(float));
    cudaMallocManaged(&resiz, len1D_new * sizeof(float));

    cudaMemcpy(raws, raw_scans, len1D_old * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    interpolationKernel << <size_to.y, size_to.x >> > (raws, resiz, size_from, size_to, z_over_xy);    // Init
    cudaDeviceSynchronize();
    cudaMemcpy(resized_scan, resiz, len1D_new * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(raws);
    cudaFree(resiz);

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    printf(" Resizing completed in %d ms.\n", duration);

    // Returning values
    *size_out = size_to;
    return resized_scan;
}