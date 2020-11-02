#pragma once
#include "Camera.h"
#include "Raytracing.h"
#include <SFML\graphics.hpp>

class Environment {

public:
	Environment() {
		camera = new Camera();
		image = new sf::Image();
		image->create(512, 512, sf::Color(0, 255, 0));
		RT.initRaytracer(camera, image);
	}

	int a = 1;
	Camera *camera;
	sf::Image *image;
	Raytracer RT;
	void Run();
private:
	bool handleEvents(sf::Event event);
	void updateSprite();
};