#pragma once
#include "Camera.h"
#include "Raytracing.h"
#include <SFML\graphics.hpp>
#include "VolumeMaker.h"

class Environment {

public:
	Environment();
	void newVolume(Block* vol);

	Camera *camera;
	sf::Image *image;
	Raytracer RT;
	VolumeMaker VM;

	Block* volume;
	void handleConsole();

	void Run();
private:
	bool handleEvents(sf::Event event);
	void updateSprite();
};