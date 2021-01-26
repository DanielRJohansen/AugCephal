#pragma once
#include "Camera.h"
#include <SFML\graphics.hpp>
#include "CudaContainers.cuh"
#include <thread>

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
	Volume* volume;
	sf::Texture* cuda_texture;

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