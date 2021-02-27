#pragma once
#include <SFML\graphics.hpp>
#include <thread>
#include <iostream>

#include "Camera.h"
#include "CudaContainers.cuh"
#include "Preprocessing.cuh"
#include "Rendering.cuh"
#include "LiveEditor.cuh"


#include "ResourceFiles\Button.h"



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
	Environment(string path, Int3 dimensions, float zoverxy, float true_voxel_volume);
	void newVolume(Block* vol);

	Camera *camera;
	sf::Image *image;
	Volume* volume;
	sf::Texture* cuda_texture;

	void handleConsole();

	void Run();
private:
	RenderEngine REE;
	Ray* rayptr_dev;				// Is live!
	LiveEditor liveeditor;

	void scheduleTask(Task);		//By subthreads
	bool handleTasks();	// By main thread
	bool volume_updated = false;
	vector<Task> tasks;

	bool handleEvents(sf::Event event);
	void handleMouseEvents(sf::Event event, sf::RenderWindow* window);
	void handleButtonEvents(sf::RenderWindow* window, sf::Sprite* sprite);

	void renderAll(sf::RenderWindow* window, sf::Sprite* sprite);



	// Features
	sf::Font MyFont;
	Button window_from;
	Button window_to;




	// For mouse actions
	sf::Vector2i prev_mousepos;
	bool left_pressed = false;
	bool right_pressed = false;

};