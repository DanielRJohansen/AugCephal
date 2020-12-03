#include "Environment.h"

const int ILLEGAL_TYPE = -33;

Environment::Environment() {
	VM = new VolumeMaker(true);	//Do clustering
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
	string cats[11] = { "lung", "fat", "fluids", "water", "hematoma", "bloodclot",
		"blood", "muscle", "cancellous", "cortical" , "foreign"};
	for (int i = 0; i < 11; i++) {
		if (cats[i] == input)
			return i;
	}
	return ILLEGAL_TYPE;
}

void Environment::handleConsole() {
	string type;
	int type_index;
	bool hide;
	string types[6] = { "lung", "fat", "fluids", "muscle", "clot", "bone" };

	while (true) {
		if (volume_updated) {
			continue;
		}
		printf("Change type visibility? (%s,%s,%s,%s,%s,%s)\n", types[0].c_str(), types[1].c_str(), 
			types[2].c_str(), types[3].c_str(), types[4].c_str(), types[5].c_str());
		cin >> type;
		printf("Hide/Show (1/0) \n");
		cin >> hide;

		// I'm sorry...
		type_index = idFromString(type);
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
