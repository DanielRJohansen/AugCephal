#include "Environment.h"

const int ILLEGAL_TYPE = -33;

Environment::Environment() {
	VM = new VolumeMaker(false);	//Do clustering
	volume = VM->volume;
	camera = new Camera();

	image = new sf::Image();
	image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
	cuda_texture = new sf::Texture;
	cuda_texture->create(RAYS_PER_DIM, RAYS_PER_DIM);

	RT.initRaytracer(camera, image);
	RT.updateVol(volume);
	RT.updateEmptySlices(VM->empty_y_slices, VM->empty_x_slices);
}


void Environment::Run() {
	cout << "Environment running" << endl;
	thread thr1;
	thr1 = thread(&Environment::handleConsole, this);

	sf::RenderWindow window(sf::VideoMode(RAYS_PER_DIM, RAYS_PER_DIM), 
		"3D body", sf::Style::Close );
	sf::Texture texture;
	sf::Sprite sprite;
	RT.render(cuda_texture);
	texture.loadFromImage(*image);
	sprite.setTexture(*cuda_texture, true);

	while (window.isOpen()) {
		window.clear();
		
		// Handle events
		sf::Event event;	// Create new each time so no event is applied twice
		if (window.pollEvent(event)) {
			if (handleEvents(event)) {	//If relevant event happened
				//texture.loadFromImage(*image);
				sprite.setTexture(*cuda_texture, true);
			}
		}
		if (handleTasks()) {
			//texture.loadFromImage(*image);
			sprite.setTexture(*cuda_texture, true);
		}
		

		window.draw(sprite);
		window.display();
	}
}


void Environment::updateSprite() {
	
}

bool Environment::handleTasks() {
	if (volume_updated) {
		RT.render(cuda_texture);
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
		action = "zoom_out";
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
	RT.render(cuda_texture);	
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
			if (VM->setIgnore(type_index, hide)) {
				printf("Updating volume...");
				RT.updateVol(VM->volume);
				volume_updated = true;
				printf(" Volume updated\n");
			}
		}
					
	}
}

void Environment::scheduleTask(Task t) {
	tasks.push_back(t);
}
