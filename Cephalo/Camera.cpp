#include "Camera.h"


void Camera::updatePos(string action) {	//char n to skip change
	if (action == "u")
		pitch += rotation_step;
	else if (action == "d")
		pitch -= rotation_step;
	else if (action == "l")
		yaw += rotation_step;
	else if (action == "r")
		yaw -= rotation_step;
	else if (action == "zoom_in")
		radius -= CAM_RADIUS_INC;
	else if (action == "zoom_out")
		radius += CAM_RADIUS_INC;
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

