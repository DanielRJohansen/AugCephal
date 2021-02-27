#include "Environment.h"

const int ILLEGAL_TYPE = -33;

Environment::Environment() {
}


Environment::Environment(Volume* vol) {
	volume = vol;

	camera = new Camera();

	image = new sf::Image();
	image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
	cuda_texture = new sf::Texture;
	cuda_texture->create(RAYS_PER_DIM, RAYS_PER_DIM);

	REE = RenderEngine(volume, camera);
}

Environment::Environment(string path, Int3 dimensions, float zoverxy, float true_voxel_volume) {


	// Handle features
	
	// Load from a font file on disk
	if (!MyFont.loadFromFile(".\\ResourceFiles\\arial.ttf"))
	{
		printf("Font not loaded wtf\n");
		exit(-1);
	}
	int btn_width = 250;
	int btn_height = 60;
	window_from = Button(RAYS_PER_DIM - btn_width - 30, 150, btn_width, btn_height, &MyFont, "From: ", HU_MIN);
	window_to = Button(RAYS_PER_DIM - btn_width - 30, 150 + btn_height + 10, btn_width, btn_height, &MyFont, "To: ", HU_MAX);



	// Prepare window
	image = new sf::Image();
	image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
	cuda_texture = new sf::Texture;
	cuda_texture->create(RAYS_PER_DIM, RAYS_PER_DIM);


	Preprocessor PP;
	volume = PP.processScan(path, dimensions, zoverxy, true_voxel_volume);

	liveeditor = LiveEditor(volume);
	volume->compactclusters = liveeditor.makeCompactClusters();// liveeditor.getCompactClusters();

	camera = new Camera();



	REE = RenderEngine(volume, camera);



}


void Environment::Run() {
	cout << "Environment running\n\n\n" << endl;
	//thread thr1;
	//thr1 = thread(&Environment::handleConsole, this);

	sf::RenderWindow window(sf::VideoMode(RAYS_PER_DIM, RAYS_PER_DIM), "3D body", sf::Style::Close );
	sf::Texture texture;
	sf::Sprite sprite;

	rayptr_dev = REE.render(cuda_texture);		// Do initial render
	liveeditor.setRayptr(rayptr_dev);			// Setup rayptr first time rendering, before polling for any clusters at pixels

	texture.loadFromImage(*image);
	sprite.setTexture(*cuda_texture, true);



	while (window.isOpen()) {
		window.clear();
		
		// Handle events
		sf::Event event;	// Create new each time so no event is applied twice
		if (window.pollEvent(event)) {
			if (handleEvents(event)) {	//If relevant event happened
				sprite.setTexture(*cuda_texture, true);
			}
		}


		handleMouseEvents(event, &window);
		handleButtonEvents(&window, &sprite);



		if (liveeditor.checkRenderFlag())
			REE.render(cuda_texture);




		renderAll(&window, &sprite);
	}
}

void Environment::renderAll(sf::RenderWindow* window, sf::Sprite* sprite)
{
	window->draw(*sprite);
	window_from.render(window);
	window_to.render(window);
	window->display();
}



void Environment::handleMouseEvents(sf::Event event, sf::RenderWindow* window) {
	sf::Vector2i mousepos = sf::Mouse::getPosition(*window);
	if (mousepos.x >= 0 && mousepos.y >= 0 && mousepos.x < RAYS_PER_DIM && mousepos.y < RAYS_PER_DIM){
		if (mousepos != prev_mousepos) {
			prev_mousepos = mousepos;
		}
	}
	else { return; }	// Ensures no actions are recorded with mouse outside window!
		

	if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left) {
		left_pressed = false;
	}
	if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Right) {
		right_pressed = false;
	}
	if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
		if (left_pressed == false) {
			int pixel_index = mousepos.y * RAYS_PER_DIM + mousepos.x;
			liveeditor.selectCluster(pixel_index);
		}
		left_pressed = true;
	}
	if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Right) {
		if (right_pressed == false) {
			liveeditor.resetClusters();
		}
		right_pressed = true;
	}
	if (event.type == sf::Event::MouseWheelMoved) {
		if (event.mouseWheel.delta < 0)
			liveeditor.hideCurrentCluster();
		else if (event.mouseWheel.delta > 0)
			liveeditor.isolateCurrentCluster();
	}
	
	
	
}

void Environment::handleButtonEvents(sf::RenderWindow* window, sf::Sprite* sprite) {
	sf::Vector2f mouspos = sf::Vector2f(sf::Mouse::getPosition(*window));
	if (window_from.update(mouspos, window, window)) {
		renderAll(window, sprite);
		window_from.inputTextLoop(window);
		liveeditor.window(window_from.getVal(), window_to.getVal());
	}
	if (window_to.update(mouspos, window, window)) {
		renderAll(window, sprite);
		window_to.inputTextLoop(window);
		liveeditor.window(window_from.getVal(), window_to.getVal());
	}


}

bool Environment::handleTasks() {
	if (volume_updated) {
		REE.render(cuda_texture);
		volume_updated = false;
		return true;
	}
}
bool Environment::handleEvents(sf::Event event) {
	string action;
	if (event.type == sf::Event::KeyPressed) {
		switch (event.key.code)
		{
		case(sf::Keyboard::Up):
			action = 'u';
			break;
		case(sf::Keyboard::Down):
			action = 'd';
			break;
		case(sf::Keyboard::Left):
			action = 'l';
			break;
		case(sf::Keyboard::Right):
			action = 'r';
			break;
		case(sf::Keyboard::I):
			action = "zoom_in";
			break;
		case(sf::Keyboard::O):
			action = "zoom_out";
			break;
		default:
			return false;		// Do nothing new with no keypress
		}
	}
	else
		return false;	// Key release events
	
	camera->updatePos(action);
	REE.render(cuda_texture);																			// CGHECJ
	return true;
}

int idFromString(string input) {
	for (int i = 0; i < NUM_CATS; i++) {
		//printf("%s          %s\n", category_names[i].c_str(), input.c_str());
		if (category_names[i] == input)
			return i;
	}
	return ILLEGAL_TYPE;
}

void Environment::handleConsole() {
	string type;
	int type_index;
	bool hide;

	while (true) {
		if (volume_updated) {
			continue;
		}
		printf("Change type visibility? (");
		for (int i = 0; i < NUM_CATS; i++) {
			printf("%s  ", category_names[i].c_str());
		}
		printf(")\n");
	
		cin >> type;
		printf("Hide/Show (1/0) \n");
		cin >> hide;

		// I'm sorry...
		type_index = idFromString(type);
		//cout << type_index << endl << endl;
		if (type_index != ILLEGAL_TYPE) {
			/*if (VM->setIgnore(type_index, hide)) {					// UHHHHHHHHHHHH
				printf("Updating volume...");
				RT.updateVol(VM->volume);
				volume_updated = true;
				printf(" Volume updated\n");
			}*/
		}
					
	}
}

void Environment::scheduleTask(Task t) {
	tasks.push_back(t);
}
