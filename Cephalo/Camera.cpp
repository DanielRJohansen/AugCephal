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
	cout << "Camera position " << x << " " << y << " " << z << endl;
}

void Camera::calcFocalPlane() {
	//focal_plane_normal = origin * (1 / radius); // Float3(-x / radius, -y / radius, -z / radius);
	//cout << "Cam yaw " << yaw / (2*3.14) * 360 << "   Cam pitch " << pitch / (2*3.14) * 360 << endl;
	//cout << "Plane yaw " << plane_yaw/ (2 * 3.14) *360 << "   plane pitch " << plane_pitch / (2 * 3.14) * 360 << endl;
	//cout << "Cam pitch yaw " << pitch << " " << yaw << endl;

	// It's a fix and it works, shut up!
	plane_pitch = 3.1415 - pitch;
	plane_yaw = yaw + 3.1415;

	
	cout << "Plane pitch yaw " << plane_pitch << " " << plane_yaw << endl;/*
	plane_pitch = acos(z / radius);
	plane_yaw = atan(y / x);
	cout << "Plane pitch yaw " << plane_pitch << " " << plane_yaw << endl;
	cout << "Focal plane normal: ";
	focal_plane_normal.print();
	cout << endl;*/
	//focal_plane_point = origin + focal_plane_normal * FOCAL_LEN;
}

