#pragma once
#include <iostream>
#include "Constants.h"
using namespace std;

class Camera {
public:
	Camera() { updatePos('n'); 	calcFocalPlane();}
	float z;
	float y;
	float x;

	float radius = 512;

	float plane_pitch; //Two planes, this is the camera plane, later the focal plane!
	float plane_yaw;

	Float3 focal_plane_point;		//x, y, z
	Float3 focal_plane_normal;	//x, y, z
	
	double rotation_step = 2 * 3.1415 / 20; // 20 clicks per rotation

	void updatePos(char key_pressed);
private:
	void cylindricToCartesian();
	void calcFocalPlane();
	float pitch = 3.1415 / 2.;
	float yaw = -3.1415 / 2.;
};

