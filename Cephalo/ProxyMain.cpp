#include <iostream>
#include "Environment.h"
#include "Raytracing.h"
#include "CudaOps.cuh"
#include <thread>


using namespace std;

class A {
public:
	float h[10] = { 0 };
};

int main() {

	Environment Env;

	Env.Run();

	return 0;
}


