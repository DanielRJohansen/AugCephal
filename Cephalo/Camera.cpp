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
	cout << "Camera position " << x << " " << y << " " << z << endl;
}

void Camera::calcFocalPlane() {
	focal_plane_normal = Float3(-x / radius, -y / radius, -z / radius);
	//cout << "Cam yaw " << yaw / (2*3.14) * 360 << "   Cam pitch " << pitch / (2*3.14) * 360 << endl;
	//cout << "Plane yaw " << plane_yaw/ (2 * 3.14) *360 << "   plane pitch " << plane_pitch / (2 * 3.14) * 360 << endl;
	cout << "Focal plane normal: ";
	focal_plane_normal.print();

	focal_plane_point = Float3(x, y, z) + focal_plane_normal * FOCAL_LEN;
}

