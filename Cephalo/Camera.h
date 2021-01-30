#pragma once
#include <iostream>
#include "Constants.h"
#include "Containers.h"
using namespace std;

class Camera {
public:
	Camera() { updatePos("n"); printf("Camera initialized at position %f %f %f\n", origin.x, origin.y, origin.z); }

	Float3 origin;
	float radius = CAM_RADIUS;

	float plane_pitch; //Two planes, this is the camera plane, later the focal plane!
	float plane_yaw;

	Float3 focal_plane_point;		//x, y, z
	Float3 focal_plane_normal;		//x, y, z
	
	double rotation_step = CAM_ROTATION_STEP;

	void updatePos(string action);
private:
	void cylindricToCartesian();
	void calcFocalPlane();
	float pitch = 0;// 3.1415 / 4.;
	float yaw = -3.1415 / 2.;
};

