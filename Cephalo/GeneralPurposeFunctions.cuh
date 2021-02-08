#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>

//#include "CudaContainers.cuh"


__device__ __host__ inline int xyzToIndex(Int3 coord, Int3 size) {
    return coord.z * size.y * size.x + coord.y * size.x + coord.x;
}

__device__ __host__ inline bool isInVolume(Int3 coord, Int3 size) {
    return coord.x >= 0 && coord.y >= 0 && coord.z >= 0 && coord.x < size.x&& coord.y < size.y&& coord.z < size.z;
}

/*
Int3 indexToXYZ(int index, Int3 size) {
    int z = index / (size.y * size.y);
    index -= z * size.y * size.x;
    int y = index / size.x;
    int x = index % size.x;
    return Int3(x, y, z);
}*/