#include "Environment.h"

#define KEY_UP 72
#define KEY_DOWN 80
#define KEY_LEFT 75
#define KEY_RIGHT 77

void Environment::Run() {
	int a = 1;
	cout << "Environment running" << endl;
	/*
	int c = 0;
	cout << "Camera rotation stepsize " << camera.rotation_step << endl;
	cout << "Camera initial position " << camera.x << " " << camera.y
		<< " " << camera.z << endl << endl;
	while (true) {
		bool move = true;
		c = 0;
		switch ((c = _getch())) {
		case KEY_UP:
			cout << endl << "Up" << endl;//key up
			camera.updatePos('u');
			break;
		case KEY_DOWN:
			cout << endl << "Down" << endl;   // key down
			camera.updatePos('d');
			break;
		case KEY_LEFT:
			cout << endl << "Left" << endl;  // key left
			camera.updatePos('l');
			break;
		case KEY_RIGHT:
			cout << endl << "Right" << endl;  // key right
			camera.updatePos('r');
			break;
		default:
			move = false;	// Only case where we dont need to render
			break;
		}
		if (move) {
			raytracer.render(camera);
		}
	}
	*/
}