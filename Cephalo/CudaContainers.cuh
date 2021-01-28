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

struct Int2 {
	Int2(int x, int y) : x(x), y(y) {}

	int x, y;
};

struct Int3 {
	__device__ __host__ Int3() {}
	__device__ __host__ Int3(int x, int y, int z) : x(x), y(y), z(z) {}

	int x, y, z;
};

struct CompactBool {
	__host__ __device__ CompactBool() {}
	__host__ unsigned* makeCompact(int size) {
		int compact_size = ceil((float)size / 16.);
		bytesize = compact_size * sizeof(unsigned);
		printf("Creating compact vector of size: %d Kb\n", bytesize / 1000);

		unsigned* compacted;
		cudaMallocManaged(&compacted, bytesize);
		return compacted;
	}
	unsigned* makeCompactHost(int size) {
		int compact_size = ceil((float)size / 16.);
		
		unsigned* compact = new unsigned[compact_size]();
		return compact;
	}

	__host__ __device__ int getIntIndex(int index) {
		return index / 16;
	}
	__host__ __device__ unsigned getBit(unsigned* compacted, int index) {
		uint8_t bit_index = index % 16;
		unsigned byte = compacted[getIntIndex(index)];
		return (byte >> bit_index) & 1U;
	}
	__host__ __device__ void setBit(unsigned* compacted, int index) {
		uint8_t bit_index = index % 16;
		printf("Index %d mapped to byte %d bit %d\n", index, getIntIndex(index), bit_index);
		compacted[getIntIndex(index)] |= 1U << bit_index;
		printf("Updated: %u\n\n", compacted[getIntIndex(index)]);
	}

	int bytesize;
	unsigned* compactedbool;		// FOR STORAGE ONLY, NO OPERATIONS ALLOWED
};


struct Voxel{	//Lots of values
	Voxel() {}	

	bool ignore = false;
	float hu_val = 10;
	int cluster_id = -1;
	int alpha = 1;
	float norm_val = 0;
	CudaColor color;

	__device__ void norm(float min, float max) {
		if (hu_val > max) { norm_val = 1; }
		else if (hu_val < min) { norm_val = 0; }
		else norm_val = (hu_val - min) / (max - min);
		color = CudaColor(norm_val);									// TEMPORARY!
	}
	void normCPU(float min, float max) {
		if (hu_val > max) { norm_val = 1; }
		else if (hu_val < min) { norm_val = 0; }
		else norm_val = (hu_val - min) / (max - min);
		color = CudaColor(norm_val);									// TEMPORARY!
	}
};

struct TestCUDA {
	TestCUDA() {};

	float a = 0;

	__device__ void dothing() {
		a = 5;
	}
};

struct TissueCluster {	// Lots of functions
	Voxel* voxels;
	float median;
};

class Volume {
public: 
	Volume(){}
	Volume(Voxel* v, Int3 size) : size(size){
		voxels = v;
		len = size.x * size.y * size.z;
	}


	Int3 size;
	int len = 0;

	// GPU side
	Voxel* voxels;	
	bool* xyColumnIgnores;
	CompactBool* xyIgnores;	//Host side
	TissueCluster* clusters;
};






