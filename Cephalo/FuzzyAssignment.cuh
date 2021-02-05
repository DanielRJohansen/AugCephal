#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <chrono>
#include "math.h"
//Dunno which of these are necessary. Maybe none?
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include "CudaContainers.cuh"
#include "GeneralPurposeFunctions.cuh"
using namespace std;

class FuzzyAssigner {


public:
	void doFuzzyAssignment(Volume* volume, int k) {
		CudaKCluster* kclusters = kMeans(volume, k, 60);			
		fuzzyClusterAssignment(volume, kclusters, k);
	}

private:
	CudaKCluster* kMeans(Volume* volume, int k, int max_iterations);
	void fuzzyClusterAssignment(Volume* volume, CudaKCluster* kclusters, int k);
};