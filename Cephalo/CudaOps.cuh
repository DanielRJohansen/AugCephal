#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Constants.h"
using namespace std;



class CudaOperator {
public:
	CudaOperator();
	void update(Ray *rp);
	void newVolume(Volume *vol) { volume = vol; }
	void rayStep();
	void doStuff();
	Ray* rayptr;
	Volume *volume;			//X, Y, Z, [color, alpha]
	int a = 0;
private:
	
	void square(int* a, int* b, int n);
};