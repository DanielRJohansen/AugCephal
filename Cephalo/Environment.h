#pragma once
#include <conio.h>
#include "Camera.h"
#include "Raytracing.h"

class Environment {

public:
	Environment() {
		RT.initRaytracer(camera); 
		//RT.render(camera);
	}

	int a = 1;
	Camera camera;
	Raytracer RT;
	void Run();
};