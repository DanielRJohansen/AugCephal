#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
//#include "Constants.h"
//#include "Containers.h"
#include <SFML\graphics.hpp>
//#include "CudaContainers.h"
#include "CudaContainers.cuh"


#include <chrono>
#include <ctime>

using namespace std;







float* Interpolate3D(float* raw_scan, Int3 size_from, Int3* size_to, float z_over_xy);	//Final arg refers to pixel spacing. Returns new size.



