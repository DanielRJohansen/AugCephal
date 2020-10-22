#include "Camera.h"
void Environment::Run() {
	int c = 0;
	cout << "Camera rotation stepsize " << camera.rotation_step << endl;
	cout << "Camera initial position " << camera.x << " " << camera.y
		<< " " << camera.z << endl << endl;
	while (true) {

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
			break;
		}
	}
}

void Camera::updatePos(char key_pressed) {	//char n to skip change
	if (key_pressed == 'u')
		pitch +- rotation_step;
	else if (key_pressed == 'd')
		pitch += rotation_step;
	else if (key_pressed == 'l')
		yaw -= rotation_step;
	else if (key_pressed == 'r')
		yaw += rotation_step;
	cylindricToCartesian();
}
void Camera::cylindricToCartesian() {	//theta, phi inverted on wiki
	//cout << pitch << "        " << yaw << endl;
	cam_plane_pitch = 
	x = radius * sin(pitch) * cos(yaw);
	y = radius * sin(pitch) * sin(yaw);
	z = radius * cos(pitch);
	//cout << x << "  " << y << "  " << z << endl << endl;;
}


