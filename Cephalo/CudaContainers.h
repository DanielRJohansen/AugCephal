#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


struct CudaColor {
	__device__ CudaColor() {};
	__device__ CudaColor(float r, float g, float b) : r(r), g(g), b(b) {};
	__device__ inline void add(CudaColor c) { r += c.r; g += c.g; b += c.b; };
	__device__ inline void cap() { if (r > 255) { r = 255; }
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

const int masksize = 5;
const int masksize3 = masksize * masksize * masksize;
struct CudaMask {
	__device__ CudaMask() {};
	__device__ CudaMask(int x, int y, int z) {
		for (int z_ = z; z_ < z + 3; z_++) {
			for (int y_ = y; y_ < y + 3; y_++) {
				for (int x_ = x; x_ < x + 3; x_++) {
					mask[xyzC(x_, y_, z_)] = 1;
				}
			}
		}
	};
	__device__ float applyMask(float kernel[masksize3]) {
		float mean = 0;
		for (int i = 0; i < masksize3; i++) {
			kernel[i] *= mask[i];
			mean += kernel[i];
		}
		mean /= 27;
		return mean;
	};				// Returns mean
	__device__ float calcVar(float kernel[masksize3], float mean) {
		float var = 0;
		for (int i = 0; i < masksize3; i++) {
			float dist = kernel[i] - mean;
			var += dist * dist;
		}
		return var;
	};		// Returns var

	float mask[masksize3] = { 0 };
	__device__ inline int xyzC(int x, int y, int z) { return z* masksize * masksize + y * masksize + x; }
};

class Test {
	__global__ Test() {};
	__device__ void dunno() {};
};

class CudaCluster {
public:
	__global__ CudaCluster() {};

	__device__ float belongingScore(float hu_val) { float dist = hu_val - mean; return dist * dist; };
	__device__ void addMember(float hu_val) { acc_hu += (long double)hu_val; };
	__device__ float getClusterMean() { return (float)acc_hu; }
	//__global__ void updateCluster() { mean = acc_hu / num_members; acc_hu = 0; num_members = 0; }
private:
	long double acc_hu = 0;
	float mean = 0;
	int num_members = 0;
};


/*struct CudaEmptyTracker {
	__global__ CudaEmptyTracker(){}
	__global__ CudaEmptyTracker(bool*x, bool* y) {}

	bool* x_empty;
	bool* y_empty;
};*/


