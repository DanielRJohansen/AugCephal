#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Constants.h"
#include "Containers.h"
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
private:
	
};