#pragma once
#include <conio.h>
#include "Camera.h"
#include "Raytracing.h"

class Environment {
	//Environment() {};// raytracer.render(camera);

public:
	int a = 1;
	Camera camera;
	Raytracer Raytracer();
	void Run();
};