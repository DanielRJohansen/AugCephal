#pragma once
#include <iostream>
#include "Camera.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Constants.h"

using namespace std;


struct RayInfo {
	RayInfo(float sp, float cp, float sy, float cy) {
		sin_pitch = sp; cos_pitch = cp; sin_yaw = sy; cos_yaw = cy; }
	float sin_pitch;
	float cos_pitch;
	float sin_yaw; 
	float cos_yaw;
};


class Ray {
	Ray() {}
public:
	float relative_pitch;
	float relative_yaw;

	Ray(float relative_pitch, float relative_yaw);
	//void castRay(float3 cam_pos, RayInfo RF);
	void rayStep(int step);
	
	~Ray() {}
private:
	//Set at beginning, never changed
	double radius = 1;	// Dunno about this value yet...................

	// New value for each new cast
	//float3 origin;
	//float3 step_vector;
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
	//float** volume;
	Ray *rayptr;

	// This optimizes cosine calculations from O(n^2) to O(n)
	float sin_pitches[RAYS_PER_DIM];
	float cos_pitches[RAYS_PER_DIM];
	float sin_yaws[RAYS_PER_DIM];
	float cos_yaws[RAYS_PER_DIM];
	

	void initRays();
	int xyToIndex(int x, int y) { return y * RAYS_PER_DIM + x; }
	//void castRays(Camera camera);	// Calculates positions, returns as list
	void catchRays();				// Determines ray rgba
	cv::Mat projectRaysOnPlane();
};