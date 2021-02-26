#include "Button.h"





Button::Button(float x, float y, float width, float height, sf::Font* font, std::string text, sf::Color idle_color, sf::Color hover_color, sf::Color active_color) {

	this->buttonState = BTN_IDLE;

	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setSize(sf::Vector2f(width, height));

	this->font = font;
	this->text.setFont(*this->font);
	this->text.setString(text);
	this->text.setFillColor(sf::Color::White);
	this->text.setCharacterSize(12);
	this->text.setPosition(
		this->shape.getPosition().x / 2.f - this->text.getGlobalBounds().width/2.f, 
		this->shape.getPosition().y/2.f - this->text.getGlobalBounds().height / 2.f
	);
	this->idle_color = idle_color;
	this->hover_color = hover_color;
	this->active_color = active_color;

	this->shape.setFillColor(this->idle_color);
}


Button::~Button() {

}



// Accessors
const bool Button::isPressed() const {
	if (this->buttonState == BTN_ACTIVE) {
		return true;
	}
	return false;
}




// Functions

void Button::update(sf::Vector2f mousepos) {
	this->buttonState = BTN_IDLE;

	if (this->shape.getGlobalBounds().contains(mousepos)) {
		this->buttonState = BTN_HOVER;

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			this->buttonState = BTN_ACTIVE;
		}
	}
	switch (this->buttonState) {
	case BTN_IDLE:
		this->shape.setFillColor(this->idle_color);
		break;
	case BTN_HOVER:
		this->shape.setFillColor(this->hover_color);
		break;
	case BTN_ACTIVE:
		this->shape.setFillColor(this->active_color);
		break;
	default:
		this->shape.setFillColor(sf::Color::Red);
		break;
	}
}

void Button::render(sf::RenderTarget* target) {
	target->draw(this->shape);

}