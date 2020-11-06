#pragma once
#include "Camera.h"
#include "Raytracing.h"
#include <SFML\graphics.hpp>

class Environment {

public:
	Environment(Block* volume) {
		camera = new Camera();
		image = new sf::Image();
		image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
		RT.initRaytracer(camera, image, volume);
	}
	void newVolume(Block* vol);

	Camera *camera;
	sf::Image *image;
	Raytracer RT;

	Block* volume;

	void Run();
private:
	bool handleEvents(sf::Event event);
	void updateSprite();
};