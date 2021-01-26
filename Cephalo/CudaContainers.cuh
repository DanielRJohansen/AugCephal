#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

struct CudaColor {
	__device__ __host__ CudaColor() {};
	__device__ __host__ CudaColor(float v) { r = v * 255; g = v * 255; b = v * 255; };
	__device__ __host__ CudaColor(float r, float g, float b) : r(r), g(g), b(b) {};
	__device__ inline void add(CudaColor c) { r += c.r; g += c.g; b += c.b; };
	__device__ inline void cap() {
		if (r > 255) { r = 255; }
		if (g > 255) { g = 255; }
		if (b > 255) { b = 255; }
	}

	__device__ inline CudaColor operator*(float s) const { return CudaColor(r * s, g * s, b * s); }
	float r = 0;
	float g = 0;
	float b = 0;
};

struct CudaFloat3 {
	__device__ CudaFloat3() {};
	__device__ CudaFloat3(float x, float y, float z) : x(x), y(y), z(z) {};
	__device__ inline CudaFloat3 operator*(float s) const { return CudaFloat3(x * s, y * s, z * s); }
	float x, y, z;
};

struct CudaRay {
	__device__ CudaRay(CudaFloat3 step_vector) : step_vector(step_vector) {};
	CudaColor color;
	CudaFloat3 step_vector;
	float alpha = 0;
};

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
	Voxel(float hu_val) : hu_val(hu_val) {}

	bool ignore = false;
	float hu_val;
	int cluster_id = -1;
	int alpha = 1;
	float norm_val;
	CudaColor color;

	void norm(int min, int max) {
		if (hu_val > max) { norm_val = 1; }
		else if (hu_val < min) { norm_val = 0; }
		else norm_val = (hu_val - min) / (max - min);
		color = CudaColor(norm_val);										// TEMPORARY!
	}
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




