#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Constants.h"
#include "Containers.h"
#include <SFML\graphics.hpp>
#include "CudaContainers.h"
#include "CudaContainers.cuh"


#include <chrono>
#include <ctime>

using namespace std;



class CudaOperator {
public:
	CudaOperator();
	void newVolume(Block* blocks);
	void updateEmptySlices(bool* yempty, bool* xempty);
	void rayStep(Ray *rp);
	void rayStepMS(Ray* rp, CompactCam cc);

	void rayStepMS(Ray* rp, CompactCam cc, sf::Texture* texture);

	Ray* rayptr;
	CompactCam* compact_cam;
	bool* dev_empty_y_slices;
	bool* dev_empty_x_slices;
	Float2 *ray_block;
	Block *blocks;			//X, Y, Z, [color, alpha]
	uint8_t* dev_image;
	uint8_t* host_image;


	// For VolumeMaker
	void medianFilter(Block *original, Block* volume);
	void rotatingMaskFilter(Block* original, Block* volume);
	void kMeansClustering(Block* volume);
private:
	int blocks_per_sm;
	int stream_size;
	int ray_stream_bytes;
	int image_stream_bytes;
};


#define WINDOW_SIZE 27
class circularWindow {
public:
	__device__ void add(int val);
	__device__ int step();

private:
	int head = 0;
	
	int window[27];
	int window_copy[27];
	int window_sorted[27];

	__device__ void sortWindow();
	__device__ void copyWindow();
	__device__ int numOutsideSpectrum();
};





float* Interpolate3D(float* raw_scan, Int3 size_from, Int3* size_to, float z_over_xy);	//Final arg refers to pixel spacing. Returns new size.



