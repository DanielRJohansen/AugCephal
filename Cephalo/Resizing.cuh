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

void doStuff();
float* Interpolate3D(float* raw_scan, Int3 size_from, Int3* size_to, float z_over_xy);	//Final arg refers to pixel spacing. Returns new size.
