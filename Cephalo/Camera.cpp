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
	calcFocalPlane();
}
void Camera::cylindricToCartesian() {	//theta, phi inverted on wiki
	float x = radius * sin(pitch) * cos(yaw);
	float y = radius * sin(pitch) * sin(yaw);
	float z = radius * cos(pitch);
	origin = Float3(x, y, z);
	//cout << "Camera position " << x << " " << y << " " << z << endl;
}

void Camera::calcFocalPlane() {
	// It's a fix and it works, shut up!
	plane_pitch = 3.1415 - pitch;
	plane_yaw = yaw + 3.1415;
}

