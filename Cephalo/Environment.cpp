#include "Environment.h"



void Environment::Run() {
	cout << "Environment running" << endl;
	
	int c = 0;
	cout << "Camera rotation stepsize " << camera->rotation_step << endl;
	cout << "Camera initial position " << camera->x << " " << camera->y
		<< " " << camera->z << endl << endl;

	sf::RenderWindow window(sf::VideoMode(512, 512), "SFML first", sf::Style::Close | sf::Style::Resize);
	sf::Texture texture;
	sf::Sprite sprite;


	while (window.isOpen()) {
		window.clear();

		// Handle events
		sf::Event event;	// Create new each time so no event is applied twice
		if (window.pollEvent(event)) {
			if (handleEvents(event)) {	//If relevant event happened
				cout << "Updating sprite " << endl;
				//for (int i = 0; i < 512; i++) { cout << (int)image->getPixel(i, i).b << " "; }
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
	switch (event.key.code)
	{
	case(sf::Keyboard::Up):
		camera->updatePos('u');
		break;
	case(sf::Keyboard::Down):
		camera->updatePos('d');
		break;
	case(sf::Keyboard::Left):
		camera->updatePos('l');
		break;
	case(sf::Keyboard::Right):
		camera->updatePos('r');
		break;
	default:
		return false;		// Do nothing new with no keypress
	}

	RT.render();	
	return true;
}