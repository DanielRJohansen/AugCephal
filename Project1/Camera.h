#pragma once
#include <iostream>

using namespace std;

class Camera {
public:
	Camera() { updatePos('n'); }
	float z;
	float y;
	float x;

	float radius = 1;

	float plane_pitch;
	float plane_yaw;

	
	double rotation_step = 2 * 3.1415 / 20; // 20 clicks per rotation

	void updatePos(char key_pressed);
private:
	void cylindricToCartesian();
	float pitch = 3.1415 / 2;
	float yaw = -3.1415 / 2;
};

