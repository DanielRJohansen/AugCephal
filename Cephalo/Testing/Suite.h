#pragma once

#include "..\Constants.h"
#include <iostream>
#include "..\ResourceFiles\Button.h"



class Suite
{
public:
	Suite();




private:
	void renderAll(sf::RenderWindow* target, sf::Sprite* sprite);





	sf::Font MyFont;
	Button window_from;
	Button window_to;

	sf::Image* image;
	sf::Texture* cuda_texture;
};

