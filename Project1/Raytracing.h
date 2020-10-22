#pragma once
#include <iostream>
#include "Camera.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
class Ray {

	public:
	//Ray(){}
	Ray(Camera camera, float relative_pitch, float relative_yaw, float stepsize);
	

	private:
	void makeStepVector(float tilt, float yaw);
	float unitvector[3][3];
	float x_origin;
	float y_origin;
	float z_origin;
	float stepsize;
	//~Ray() {}

};

class Raytracer {

public:
	cv::Mat render(Camera camera);
	float* object;

};