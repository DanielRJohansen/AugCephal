#pragma once
#include <iostream>
#include "Camera.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Constants.h"
#include "Cudaops.cuh"

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
	float* makeStepVector(RayInfo RF);	//Returns vector dx dy dz

	~Ray() {}
private:
	//Set at beginning, never changed
	double radius = 1;	// Dunno about this value yet...................
	float* step_vector = new float[3];
};


class Raytracer {
public:
	Raytracer() {};
	void initRaytracer(Camera camera);
	int a;
	//void newVolume(float** vol) { volume = vol; }
	cv::Mat render(Camera camera);
	~Raytracer();

private:
	//float** volume;
	Ray *rayptr;
	Camera camera;

	// This optimizes cosine calculations from O(n^2) to O(n)
	float sin_pitches[RAYS_PER_DIM];
	float cos_pitches[RAYS_PER_DIM];
	float sin_yaws[RAYS_PER_DIM];
	float cos_yaws[RAYS_PER_DIM];
	

	float*** all_step_vectors;		//RAYy, RAYx, RAYstepvector(x, y, z)
	float* origin = new float[3];	//x, y, z
	CudaOperator CudaOps;

	void initRays();
	void initCuda();
	void updateCameraOrigin();
	int xyToIndex(int x, int y) { return y * RAYS_PER_DIM + x; }
	void precalcSinCos();
	void castRays();	// Calculates positions, returns as list
	void catchRays();				// Determines ray rgba
	cv::Mat projectRaysOnPlane();
};