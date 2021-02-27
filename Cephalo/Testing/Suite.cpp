#include "Suite.h"




Suite::Suite() {



	// Load from a font file on disk
	if (!MyFont.loadFromFile(".\\ResourceFiles\\arial.ttf"))
	{
		printf("Font not loaded wtf\n");
		exit(-3);
	}




	// Prepare window
	image = new sf::Image();
	image->create(RAYS_PER_DIM, RAYS_PER_DIM, sf::Color(0, 255, 0));
	cuda_texture = new sf::Texture;
	cuda_texture->create(RAYS_PER_DIM, RAYS_PER_DIM);



	sf::RenderWindow window(sf::VideoMode(RAYS_PER_DIM, RAYS_PER_DIM), "3D body", sf::Style::Close);
	sf::Texture texture;
	sf::Sprite sprite;

	
	uint8_t* im = new uint8_t[NUM_RAYS * 4];
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			im[y * RAYS_PER_DIM * 4 + x * 4 + 0] = 255. / RAYS_PER_DIM * x;
			im[y * RAYS_PER_DIM * 4 + x * 4 + 1] = 0;
			im[y * RAYS_PER_DIM * 4 + x * 4 + 2] = 0;
			im[y * RAYS_PER_DIM * 4 + x * 4 + 3] = 255;
		}
	}
	cuda_texture->update(im);

	texture.loadFromImage(*image);
	sprite.setTexture(*cuda_texture, true);


	window_from = Button(500, 500, 200, 50, &MyFont, "Window From", sf::Color::Green, sf::Color::Magenta, sf::Color::Blue);
	window_to = Button(500, 600, 200, 50, &MyFont, "Window To", sf::Color::Green, sf::Color::Magenta, sf::Color::Blue);



	while (window.isOpen()) {
		window.clear();

		// Handle events
		sf::Event event;	// Create new each time so no event is applied twice

		sf::Vector2f mouspos = sf::Vector2f(sf::Mouse::getPosition(window));
		window_from.update(mouspos);
		window_to.update(mouspos);
		
		window.draw(sprite);
		window_from.render(&window);
		window_to.render(&window);
		window.display();
	}
}