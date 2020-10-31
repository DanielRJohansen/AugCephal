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





class Raytracer {
public:
	Raytracer() {};
	void initRaytracer(Camera camera);
	int a;
	Volume volume;
	cv::Mat render(Camera camera);
	~Raytracer();

private:
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
	void initRenderPlane();
	
	int xyToRayIndex(int x, int y) { return y * RAYS_PER_DIM + x; }
	int rayIndexToX(int index) { return index % RAYS_PER_DIM; }
	int rayIndexToY(int index) { return index / RAYS_PER_DIM; }

	void updateCameraOrigin();
	void precalcSinCos();
	void castRays();	// Calculates positions, returns as list
	void catchRays();				// Determines ray rgba
	cv::Mat projectRaysOnPlane();
};