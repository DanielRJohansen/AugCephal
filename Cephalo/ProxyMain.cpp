#include <iostream>
#include "Environment.h"
#include "Raytracing.h"
#include "CudaOps.cuh"
#include <thread>
#include "Containers.h"
#include "SliceMagic.h"
using namespace std;

class A {
public:
	float h[10] = { 0 };
};

void test(int* hllo) {
	hllo = new int[300000];
	hllo[30] = 99;
}


int main() {


	SliceMagic SliceM;

	Environment Env;
	Env.Run();

	return 0;
}


