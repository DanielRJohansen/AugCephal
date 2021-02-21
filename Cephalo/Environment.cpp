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
Environment::Environment(string path, Int3 dimensions, float zoverxy) {
	Preprocessor PP;
	volume = PP.processScan(path, dimensions, zoverxy);

	liveeditor = LiveEditor(volume);
	volume->compactclusters = liveeditor.getCompactClusters();

	camera = new Camera();

	image = new sf::Image();
	image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
	cuda_texture = new sf::Texture;
	cuda_texture->create(RAYS_PER_DIM, RAYS_PER_DIM);

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
		/*if (handleTasks()) {
			sprite.setTexture(*cuda_texture, true);
		}*/

		handleMouseEvents(event, &window);


		

		window.draw(sprite);
		window.display();
	}
}


void Environment::handleMouseEvents(sf::Event event, sf::RenderWindow* window) {
	sf::Vector2i mousepos = sf::Mouse::getPosition(*window);
	if (mousepos.x >= 0 && mousepos.y >= 0 && mousepos.x < RAYS_PER_DIM && mousepos.y < RAYS_PER_DIM){
		if (mousepos != prev_mousepos) {
			printf("\rMouse pos: x: %04d y: %04d", mousepos.x, mousepos.y);
			prev_mousepos = mousepos;
		}
	}
	else { return; }	// Ensures no actions are recorded with mouse outside window!
		

	if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left) {
		left_pressed = false;
	}
	if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
		if (left_pressed == false) {
			printf("  click  ");
			int pixel_index = mousepos.y * RAYS_PER_DIM + mousepos.x;
			liveeditor.selectCluster(pixel_index);
		}
		left_pressed = true;
	}

	if (event.type == sf::Event::MouseWheelMoved) {
		printf("   D:%d ", event.mouseWheel.delta);
	}
}


void Environment::updateSprite() {
	
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

	/*if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
		action = "zoom_in";
	}
	else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
		action = "zoom_out";sfml get g
	}*/

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
