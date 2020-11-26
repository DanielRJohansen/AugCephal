#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


struct CudaColor {
	__device__ CudaColor() {};
	float r = 0;
	float b = 0;
	float g = 0;
};

struct CudaFloat3 {
	__device__ CudaFloat3() {};
	__device__ CudaFloat3(float x, float y, float z) : x(x), y(y), z(z) {};
	float x, y, z;
	__device__ inline CudaFloat3 operator*(float s) const { return CudaFloat3(x * s, y * s, z * s); }
};

class CudaRay {
	__device__ CudaRay(CudaFloat3 step_vector) : step_vector(step_vector) {};
	CudaColor color;
	CudaFloat3 step_vector;
	float alpha;
};

