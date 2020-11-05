#include "Environment.h"



void Environment::Run() {
	cout << "Environment running" << endl;
	cout << "Camera rotation stepsize " << camera->rotation_step << endl;


	sf::RenderWindow window(sf::VideoMode(512, 512), "SFML first", sf::Style::Close | sf::Style::Resize);
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
	char chr;
	if (event.type == sf::Event::KeyPressed) {
		switch (event.key.code)
		{
		case(sf::Keyboard::Up):
			chr = 'u';
			break;
		case(sf::Keyboard::Down):
			chr = 'd';
			break;
		case(sf::Keyboard::Left):
			chr = 'l';
			break;
		case(sf::Keyboard::Right):
			chr = 'r';
			break;
		default:
			return false;		// Do nothing new with no keypress
		}
	}
	else
		return false;	// Key release events
	
	camera->updatePos(chr);
	RT.render();	
	return true;
}