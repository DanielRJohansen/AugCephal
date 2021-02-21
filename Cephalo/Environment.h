#pragma once
#include <SFML\graphics.hpp>
#include <thread>
#include <iostream>

#include "Camera.h"
#include "CudaContainers.cuh"
#include "Preprocessing.cuh"
#include "Rendering.cuh"
#include "LiveEditor.cuh"




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
	Environment(string path, Int3 dimensions, float zoverxy);
	void newVolume(Block* vol);

	Camera *camera;
	sf::Image *image;
	Volume* volume;
	sf::Texture* cuda_texture;

	void handleConsole();

	void Run();
private:
	RenderEngine REE;
	LiveEditor liveeditor;

	void scheduleTask(Task);		//By subthreads
	bool handleTasks();	// By main thread
	bool volume_updated = false;
	vector<Task> tasks;

	bool handleEvents(sf::Event event);
	void handleMouseEvents(sf::Event event, sf::RenderWindow* window);
	void updateSprite();









	// For mouse actions
	sf::Vector2i prev_mousepos;
	bool left_pressed = false;

};