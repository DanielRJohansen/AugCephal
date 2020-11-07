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



int main() {

	VolumeMaker VM;

	Environment Env(VM.volume);
	Env.Run();
	//Ray R(camera, 3.14 * 0.5, 3.14*0., 1.);

	return 0;

}


