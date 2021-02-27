#ifndef BUTTON_H
#define BUTTON_H


#include<iostream>
#include<ctime>
#include<cstdlib>
#include<sstream>

#include"SFML\System.hpp"
#include"SFML\Window.hpp"
#include "SFML\Graphics.hpp"


enum button_states { BTN_IDLE = 0, BTN_HOVER, BTN_ACTIVE };

class Button
{
private:
	short unsigned buttonState;
	bool pressed;
	bool hover;
	sf::RectangleShape shape;
	sf::Font* font;
	sf::Text text;

	sf::Color idle_color;
	sf::Color hover_color;
	sf::Color active_color;

public:
	Button(){}
	Button(float x, float y, float width, float height, sf::Font* font, std::string text, 
		sf::Color idle_color, sf::Color hover_color, sf::Color active_color);
	~Button();


	// Accessors
	const bool isPressed() const;


	void updateText(std::string value);
	void update(sf::Vector2f mousepos);
	void render(sf::RenderTarget* target);
};













#endif