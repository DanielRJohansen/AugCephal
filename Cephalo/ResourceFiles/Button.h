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


	// My additions
	int default_val;
	int value;
	std::string og_text;
public:
	Button(){}
	Button(float x, float y, float width, float height, sf::Font* font, std::string text, int default_val);
	~Button();


	// Accessors
	inline int getVal() { return value; }


	void updateText(std::string value);
	void inputTextLoop(sf::Window* window);
	bool update(sf::Vector2f mousepos, sf::Window* window, sf::RenderTarget* target);	// Returns true when Env must call inputTextLoop
	void reset();
	void render(sf::RenderTarget* target);
};













#endif