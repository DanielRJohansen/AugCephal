#pragma once
#include <conio.h>
#include "Camera.h"
#include "Raytracing.h"

class Environment {

public:
	Environment() {
		camera = new Camera();
		RT.initRaytracer(camera); 
		cv::Mat first_im = RT.render();
		cv::imshow("Image", first_im);
		//cv::waitKey();
	}

	int a = 1;
	Camera *camera;
	Raytracer RT;
	void Run();
};