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
	int size = 27;
	float best = 0;

	__device__ void add(float val);
	__device__ float step();

private:
	int head = 0;
	
	float window[27];
	float window_copy[27];
	float window_sorted[27];

	__device__ void sortWindow();
	__device__ void copyWindow();
};