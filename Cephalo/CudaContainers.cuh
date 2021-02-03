#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
using namespace std;

struct Int2 {
	__device__ __host__ Int2() {};
	__device__ __host__ Int2(int x, int y) : x(x), y(y) {}

	int x, y;
};
struct CudaFloat3;

struct Int3 {
	__device__ __host__ Int3() {}
	__device__ Int3(CudaFloat3 s);
	__device__ __host__ Int3(int x, int y, int z) : x(x), y(y), z(z) {}

	int x, y, z;
};

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

	__device__ __host__ CudaColor getRandColor() { return CudaColor(rand() % 255, rand() % 255, rand() % 255); }



	__device__ inline CudaColor operator*(float s) const { return CudaColor(r * s, g * s, b * s); }
	float r = 0;
	float g = 0;
	float b = 0;
};

struct CudaFloat3 {
	__device__ CudaFloat3() {};
	__device__ CudaFloat3(Int3 s) { x = s.x; y = s.y; z = s.z; };
	__device__ CudaFloat3(float x, float y, float z) : x(x), y(y), z(z) {};
	__device__ inline CudaFloat3 operator*(float s) const { return CudaFloat3(x * s, y * s, z * s); }
	__device__ inline CudaFloat3 operator+(CudaFloat3 s) const { return CudaFloat3(x + s.x, y + s.y, z + s.z); }

	__device__ inline float len() { sqrt(x * x + y * y + z * z); }
	__device__ void norm() {
		float s = 1/len();
		x *= s;
		y *= s;
		z *= s;
	}
	float x, y, z;
};

struct CudaRay {
	__device__ CudaRay(CudaFloat3 step_vector) : step_vector(step_vector) {};
	CudaColor color;
	CudaFloat3 step_vector;
	float alpha = 0;
};



struct CompactBool {
	__host__ __device__ CompactBool() {}
	__host__ CompactBool(bool* boolean_gpu, int column_len) {
		int compact_size = ceil((float)column_len / 32.);
		int compact_bytesize = compact_size * sizeof(unsigned);
		int boolbytesize = column_len * sizeof(bool);
		printf("Creating compact vector for %d values of size: %d Kb\n", column_len, compact_bytesize / 1000);

		bool* boolean_host = new bool[column_len];
		cudaMemcpy(boolean_host, boolean_gpu, boolbytesize, cudaMemcpyDeviceToHost);

		unsigned* compact = new unsigned[compact_size]();
		for (int i = 0; i < column_len; i++) {
			if (boolean_host[i])
				setBit(compact, i);
		}
		cudaMallocManaged(&compact_gpu, compact_bytesize);
		cudaMemcpy(compact_gpu, compact, compact_bytesize, cudaMemcpyHostToDevice);
		//delete(compact, boolean_host);	
		compact_host = compact;
		cudaFree(boolean_gpu);
	}

	__host__ __device__ unsigned getBit(unsigned* compacted, int index) {
		unsigned byte = compacted[quadIndex(index)];
		return (byte >> bitIndex(index)) & (unsigned)1;
	}
	__device__ unsigned getQuad(unsigned* compacted, int index) {
		unsigned quad = compacted[quadIndex(index)];
		return quad;
	}
	__host__ __device__ inline int quadIndex(int index) { return index / 32; }

	unsigned* compact_gpu;		// FOR STORAGE ONLY, NO OPERATIONS ALLOWED
	unsigned* compact_host;
private:
	__host__ __device__ void setBit(unsigned* compacted, int index) {
		compacted[quadIndex(index)] |= ((unsigned)1 << bitIndex(index));
	}
	__host__ __device__ inline int bitIndex(int index) { return index % 32; }
};


struct Voxel{	//Lots of values
	Voxel() {}	

	bool ignore = false;
	float hu_val = 10;
	int cluster_id = -1;
	float alpha = 1;
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
	CompactBool* CB;	//Host side
	TissueCluster* clusters;
};

const int OUTSIDEVOL = -9393;
const int ISIGNORE = -9394;

struct CudaMask {
	__host__ __device__ CudaMask() {}
	__host__ __device__ CudaMask(int uselessval) {
		for (int i = 0; i < 125; i++)
			mask[i] = 0;
	}
	__host__ __device__ CudaMask(int x, int y, int z) {
		for (int i = 0; i < masksize; i++) {
			mask[i] = 0;
		}
		for (int z_ = z; z_ < z + 3; z_++) {
			for (int y_ = y; y_ < y + 3; y_++) {
				for (int x_ = x; x_ < x + 3; x_++) {
					mask[xyzC(x_, y_, z_)] = 1;
				}
			}
		}
	}
	

	__device__ float applyMask(float* kernel) {
		mean = 0;
		penalty = 1;
		weight = 0;
		for (int i = 0; i < 125; i++) {			
			masked_kernel[i] = kernel[i] * mask[i];

			if (masked_kernel[i] == OUTSIDEVOL) {
				return 9999;
			}
			else if (masked_kernel[i] == ISIGNORE) {
				penalty += 0.5;
			}
			else {
				mean += masked_kernel[i];
				weight += mask[i];
			}
				
		}

		if (weight < 3)
			return 9999;
		
			
			
		mean /= (weight);
		return calcVar(mean);
	};
	__device__ float calcVar(float mean) {
		float var = 0;
		for (int i = 0; i < 125; i++) {
			float dist = masked_kernel[i] - mean;
			var += abs(dist) *  mask[i];// *dist;
		}
		
		return var / weight * penalty;
	};

	__host__ __device__ inline int xyzC(int x, int y, int z) { return z * 5 * 5 + y * 5 + x; }
	
	// Constant
	int masksize = 125;
	float mask[125];

	// Changed each step
	float weight;
	float penalty;
	float mean;
	float masked_kernel[125];
};

class TestClass {
public:
	__host__ __device__ TestClass() {}
	__host__ __device__ ~TestClass() {}
	int id;
	float val;
};

class CudaKCluster {
public:
	__host__ __device__ CudaKCluster() {}
	__host__ __device__ CudaKCluster(int id): id(id) {}
	__host__ __device__ ~CudaKCluster() {}

	float calcCentroid() { 
		float old = centroid;
		if (num_members == 0)
			num_members = 1;
		centroid = (float)(accumulation / num_members); 
		prev_members = num_members;
		num_members = 0;
		accumulation = 0;
		printf("    K-Cluster %02d	centroid: %f    members: %d\n", id, centroid, prev_members);
		return abs(old - centroid);
	}
	__device__ float inline belonging(float val) { return 1-abs(val - centroid); }
	__device__ void assign(float val) { 
		while (blocked) {}
		blocked = true;
		num_members+=1; 
		accumulation += val; 
		blocked = false;
	}
	void assignBatch(float acc, int cnt) {
		accumulation += acc;
		num_members += cnt;
	}
	void mergeBatch(CudaKCluster c) {
		num_members += c.num_members;
		accumulation += c.accumulation;
	}

	bool blocked = false;

	int num_members = 0;
	int prev_members= -1;
	float accumulation = 0;

	int id;
	float centroid = -1;
	CudaColor color = CudaColor().getRandColor();
};
/*
struct CudaMask2 {
	__device__ CudaMask2() {}
	__device__ CudaMask2(int a) {
		for (int i = 0; i < 125; i++) {
			mask[i] = 0;
		}
	}
	__device__ CudaMask2(int x, int y, int z) {
		for (int i = 0; i < masksize; i++) {
			mask[i] = 0;
		}
		for (int z_ = z; z_ < z + 3; z_++) {
			for (int y_ = y; y_ < y + 3; y_++) {
				for (int x_ = x; x_ < x + 3; x_++) {
					mask[xyzC(x_, y_, z_)] = 1;
				}
			}
		}
	}
	float mask[125];
	const int masksize = 125;
	__device__ inline int xyzC(int x, int y, int z) { return z * 125 * 125 + y * 125 + x; }
};
	__device__ float applyMask(float kernel[125]) {
		float mean = 0;
		for (int i = 0; i < 125; i++) {
			kernel[i] *= mask[i];
			mean += kernel[i];
		}
		mean /= 27;
		return mean;
	};				// Returns mean
		// Returns var
	*/




