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

	plane_pitch = pitch + 3.1415;
	plane_yaw = yaw + 3.1415;
	cylindricToCartesian();
}
void Camera::cylindricToCartesian() {	//theta, phi inverted on wiki
	x = radius * sin(pitch) * cos(yaw);
	y = radius * sin(pitch) * sin(yaw);
	z = radius * cos(pitch);
}

void Camera::calcFocalPlane() {
	focal_plane_normal = Float3(sin(plane_pitch) * cos(plane_yaw),
		sin(plane_pitch) * cos(plane_yaw), cos(plane_pitch));
	//focal_plane_normal[0] = 1 * sin(plane_pitch) * cos(plane_yaw);
	//focal_plane_normal[1] = 1 * sin(plane_pitch) * sin(plane_yaw);
	//focal_plane_normal[2] = 1 * cos(plane_pitch);

	focal_plane_point = Float3(x, y, z) + focal_plane_normal * FOCAL_LEN;
	//focal_plane_point[0] = x + focal_plane_normal[0] * FOCAL_LEN;
	//focal_plane_point[1] = y + focal_plane_normal[1] * FOCAL_LEN;
	//focal_plane_point[2] = z + focal_plane_normal[2] * FOCAL_LEN;
}


