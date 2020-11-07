#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Constants.h"
#include "Containers.h"
#include <ctime>

using namespace std;


struct testObject {
	float var = 0.42;
};

class CudaOperator {
public:
	CudaOperator();
	void newVolume(Block* blocks);
	void rayStep(Ray *rp);
	Ray* rayptr;
	Block *blocks;			//X, Y, Z, [color, alpha]
	

	// For VolumeMaker
	void medianFilter(Block *original, Block* volume);
private:
	
};

class circularWindow {
public:
	__device__ circularWindow();
	__device__ void addTop(float* top);
	__device__ float step(float* top);

private:
	int head = 0;
	int size = 27;
	
	float *window;
	float* window_copy;
	float* window_sorted;

	__device__ void sortWindow();
	__device__ void copyWindow();
};