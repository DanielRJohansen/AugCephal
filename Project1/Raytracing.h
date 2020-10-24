#pragma once
#include <iostream>
#include "Camera.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Constants.h"

using namespace std;


struct RayInfo {
	float sin_pitch;
	float cos_pitch;
	float sin_yaw; 
	float cos_yaw;
};


class Ray {

public:
	Ray(){}
	Ray(float relative_pitch, float relative_yaw, float stepsize);
	void castRay(float x, float y, float z, RayInfo RF);
	float relative_pitch;
	float relative_yaw;
	//~Ray() {}
private:
	//Set at beginning, never changed
	float stepsize;
	double radius = 1;	// Dunno about this value yet...................

	// New value for each new cast
	float x_origin;
	float y_origin;
	float z_origin;

	void makeStepVector(RayInfo RF);
	float unitvector[3][3];

	

};


class Raytracer {
public:
	Raytracer();
	int a;
	//void newVolume(float** vol) { volume = vol; }
	cv::Mat render(Camera camera);
	~Raytracer();
private:
	float** volume;

	Ray rays[RAYS_PER_DIM][RAYS_PER_DIM];
	// This optimizes cosine calculations from O(n^2) to O(n)
	float sin_pitches[RAYS_PER_DIM];
	float cos_pitches[RAYS_PER_DIM];
	float sin_yaws[RAYS_PER_DIM];
	float cos_yaws[RAYS_PER_DIM];

	void initRays();
	void castRays(Camera camera);	// Calculates positions, returns as list
	void catchRays();				// Determines ray rgba
	cv::Mat projectRaysOnPlane();
};