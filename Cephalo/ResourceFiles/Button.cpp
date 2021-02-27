#include "Button.h"





Button::Button(float x, float y, float width, float height, sf::Font* font, std::string text, int default_val) {

	this->buttonState = BTN_IDLE;

	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setSize(sf::Vector2f(width, height));

	this->font = font;
	this->text.setFont(*this->font);
	this->text.setString(text);
	this->text.setFillColor(sf::Color::Black);
	this->text.setCharacterSize(30);
	this->text.setPosition(
		this->shape.getPosition().x + shape.getGlobalBounds().width * 0.5  - this->text.getGlobalBounds().width / 2.f, 
		this->shape.getPosition().y + shape.getGlobalBounds().height * 0.5 - this->text.getGlobalBounds().height / 2.f -3
	);
	this->idle_color = sf::Color(0x6096BAD0);
	this->hover_color = sf::Color(0x6096BAFF);
	this->active_color = sf::Color(0x274C77FF);

	this->shape.setFillColor(this->idle_color);






	this->default_val = default_val;
	this->value = default_val;
	og_text = text;
	updateText(std::to_string(default_val));
}


Button::~Button() {

}







// Functions

void Button::updateText(std::string value) {


	this->text.setString(og_text + value);
	this->text.setPosition(
		this->shape.getPosition().x + shape.getGlobalBounds().width * 0.5 - this->text.getGlobalBounds().width / 2.f,
		this->shape.getPosition().y + shape.getGlobalBounds().height * 0.5 - this->text.getGlobalBounds().height / 2.f - 3
	);
}

void Button::inputTextLoop(sf::Window* window) {
	sf::Event event;
	std::string inputstring = "";
	while (true) {
		while (window->pollEvent(event)) {
			if (event.type == sf::Event::TextEntered)
			{
				if (event.text.unicode == 13)
					goto Finish;
				if (event.text.unicode > 47 && event.text.unicode < 58 || (event.text.unicode == 45 && inputstring == "") ) {
					std::cout << "ASCII character typed: " << static_cast<char>(event.text.unicode) << std::endl;
					inputstring = inputstring + static_cast<char>(event.text.unicode);
					updateText(inputstring);
				}
			}
		}
	}
Finish:
	if (inputstring != "")
		value = std::stoi(inputstring);
}

bool Button::update(sf::Vector2f mousepos, sf::Window* window, sf::RenderTarget* target) {
	this->buttonState = BTN_IDLE;

	if (this->shape.getGlobalBounds().contains(mousepos)) {
		this->buttonState = BTN_HOVER;

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			this->buttonState = BTN_ACTIVE;
		}
	}

	std::string value;
	switch (this->buttonState) {
	case BTN_IDLE:
		this->shape.setFillColor(this->idle_color);
		break;
	case BTN_HOVER:
		this->shape.setFillColor(this->hover_color);
		break;
	case BTN_ACTIVE:
		this->shape.setFillColor(this->active_color);
		return true;
	default:
		this->shape.setFillColor(sf::Color::Red);
		break;
	}
	return false;
}

void Button::reset() {
	this->value = default_val;
	updateText(std::to_string(this->value));
}

void Button::render(sf::RenderTarget* target) {
	target->draw(this->shape);
	target->draw(this->text);
}