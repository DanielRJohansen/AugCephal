#include <iostream>
#include "Environment.h"
#include "Raytracing.h"
#include "CudaOps.cuh"
#include "VolumeMaker.h"
//#include "Camera.h"
//#include "Tools.h"
//#include <opencv2/imgproc/imgproc.hpp>
//#include <cv.h>
using namespace std;

class A {
public:
	float h[10] = { 0 };
};

int main() {

	VolumeMaker VM;
	Environment Env(VM.volume);
	Env.Run();

	return 0;
}


