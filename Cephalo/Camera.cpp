#include "Camera.h"


void Camera::updatePos(char key_pressed) {	//char n to skip change
	if (key_pressed == 'u')
		pitch -= rotation_step;
	else if (key_pressed == 'd')
		pitch += rotation_step;
	else if (key_pressed == 'l')
		yaw -= rotation_step;
	else if (key_pressed == 'r')
		yaw += rotation_step;
	cylindricToCartesian();
}
void Camera::cylindricToCartesian() {	//theta, phi inverted on wiki
	plane_pitch = pitch + 3.1415;
	plane_yaw = yaw + 3.1415;
	x = radius * sin(pitch) * cos(yaw);
	y = radius * sin(pitch) * sin(yaw);
	z = radius * cos(pitch);
	//cout << pitch << "        " << yaw << endl;
	//cout << plane_pitch << "        " << plane_yaw << endl;
	//cout << endl;
}


