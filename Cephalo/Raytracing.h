#pragma once
#include <iostream>
#include "Camera.h"
#include <SFML\graphics.hpp>
#include "Constants.h"
#include "Containers.h"
#include "Cudaops.cuh"
#include "math.h"
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
	void initRaytracer(Camera *camera, sf::Image *im, Block* vol);
	void newVolume(Block* volume) { CudaOps.newVolume(volume); }

	sf::Image *image;	//

	void updateVol(Block* vol) { CudaOps.newVolume(vol); }
	void render();

	~Raytracer();

private:
	Ray *rayptr;
	Camera *camera;

	// This optimizes cosine calculations from O(n^2) to O(n)
	float sin_pitches[RAYS_PER_DIM];
	float cos_pitches[RAYS_PER_DIM];
	float sin_yaws[RAYS_PER_DIM];
	float cos_yaws[RAYS_PER_DIM];
	
	CudaOperator CudaOps;
	ColorScheme colorscheme;

	void initRays();
	
	int xyToRayIndex(int x, int y) { return y * RAYS_PER_DIM + x; }
	int rayIndexToX(int index) { return index % RAYS_PER_DIM; }
	int rayIndexToY(int index) { return index / RAYS_PER_DIM; }

	void castRays();	// Calculates positions, returns as list
	void catchRays();				// Determines ray rgba, not implemented yet
	void projectRaysOnPlane();
};