#include "Environment.h"


Environment::Environment() {
	VolumeMaker VM(false);
	volume = VM.volume;
	camera = new Camera();
	image = new sf::Image();
	image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
	RT.initRaytracer(camera, image, volume);
}


void Environment::Run() {
	cout << "Environment running" << endl;
	thread thr1;
	thr1 = thread(&Environment::handleConsole, this);

	sf::RenderWindow window(sf::VideoMode(RAYS_PER_DIM, RAYS_PER_DIM), 
		"3D body", sf::Style::Close | sf::Style::Resize);
	sf::Texture texture;
	sf::Sprite sprite;
	RT.render();
	texture.loadFromImage(*image);
	sprite.setTexture(texture, true);



	while (window.isOpen()) {
		window.clear();

		// Handle events
		sf::Event event;	// Create new each time so no event is applied twice
		if (window.pollEvent(event)) {
			if (handleEvents(event)) {	//If relevant event happened
				texture.loadFromImage(*image);
				sprite.setTexture(texture, true);
			}
		}


		window.draw(sprite);
		window.display();
	}
}


void Environment::updateSprite() {
	
}


bool Environment::handleEvents(sf::Event event) {
	string action;
	if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
		action = "zoom_in";
	}
	else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
		action = "zoom_out";
	}
	else if (event.type == sf::Event::KeyPressed) {
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
		default:
			return false;		// Do nothing new with no keypress
		}
	}
	else
		return false;	// Key release events
	
	camera->updatePos(action);
	RT.render();	
	return true;
}

void Environment::handleConsole() {
	string type;
	string types[6] = { "lung", "fat", "fluids", "muscle", "clot", "bone" };

	while (true) {
		printf("Change type visibility? (%s,%s,%s,%s,%s,%s)\n", types[0].c_str(), types[1].c_str(), 
			types[2].c_str(), types[3].c_str(), types[4].c_str(), types[5].c_str());
		cin >> type;
		cout << type;
	}
}