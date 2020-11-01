#pragma once
#include <conio.h>
#include "Camera.h"
#include "Raytracing.h"

class Environment {

public:
	Environment() {
		RT.initRaytracer(camera); 
		cv::Mat first_im = RT.render(camera);
		cv::imshow("Image", first_im);
		cv::waitKey();
	}

	int a = 1;
	Camera camera;
	Raytracer RT;
	void Run();
};