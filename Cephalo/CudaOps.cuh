#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

class CudaOperator {
public:
	CudaOperator() {};
	void updateValues(float*** sv, float* o) { all_step_vectors = sv; origin = 0; cout << "Cuda initialized" << endl; };
	void doStuff();
	float*** all_step_vectors;
	float* origin;
	int a = 0;
private:
	
	void square(int* a, int* b, int n);
};