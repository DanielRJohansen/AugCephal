#pragma once
#include "Camera.h"
#include "Raytracing.h"
#include <SFML\graphics.hpp>
#include "VolumeMaker.h"
#include "CudaContainers.cuh"

struct Task {
	Task(int c, bool h) { cat_index = c; hide = h; }
	int cat_index;
	bool hide;
};

class Environment {

public:
	Environment();
	Environment(Volume* volume);
	Environment(Block* vol);
	void newVolume(Block* vol);

	Camera *camera;
	sf::Image *image;
	Raytracer RT;
	VolumeMaker *VM;

	sf::Texture* cuda_texture;

	Block* volume;
	void handleConsole();

	void Run();
private:
	void scheduleTask(Task);		//By subthreads
	bool handleTasks();	// By main thread
	bool volume_updated = false;
	vector<Task> tasks;

	bool handleEvents(sf::Event event);
	void updateSprite();
};