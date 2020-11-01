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
	void doStuff();
	void objectTesting(testObject *t);
	testObject *t;
	Ray* rayptr;
	Block *blocks;			//X, Y, Z, [color, alpha]
	int a = 0;
private:
	
	void square(int* a, int* b, int n);
};