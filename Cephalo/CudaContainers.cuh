#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

__global__ struct Int2 {
	Int2(int x, int y) : x(x), y(y) {}

	int x, y;
};

struct Int3 {
	__device__ __host__ Int3() {}
	__device__ __host__ Int3(int x, int y, int z) : x(x), y(y), z(z) {}


	int x, y, z;
};

struct Voxel{	//Lots of values
	Voxel() {}
	Voxel(int hu_val) : hu_val(hu_val) {}

	int hu_val;
	int cluster_id = -1;
	int alpha;
	float norm_val;
};

struct TissueCluster {	// Lots of functions
	Voxel* voxels;
	float median;
};

class Volume {
public: 
	Volume(){}
	Volume(Int3 size, float* scan) : size(size) {
		len = size.x * size.y * size.z;
		voxels = new Voxel[len];
		for (int i = 0; i < len; i++) {
			//printf("%d: %f\n"i, scan[i]);
			voxels[i] = Voxel((int)scan[i]);
		}
	};

	Int3 size;
	int len;
	Voxel* voxels;
	TissueCluster* clusters;
};