#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <SFML\graphics.hpp>
#include "CudaContainers.cuh"
#include "Containers.h"
#include "Constants.h"
#include <chrono>
#include <ctime>
#include "Camera.h"



__device__ const float RAY_SS = 1;
__device__ const float e = 2.7182;

__device__ 

class RenderEngine {
public:
	RenderEngine() {};
	RenderEngine(Volume* vol, Camera* c) {

		volume = vol;
		camera = c;

		CUDAPlanning();

		cudaMallocManaged(&voxels, vol->len * sizeof(Voxel));
		updateVolume();

		rayptr_host = initRays();
		cudaMallocManaged(&rayptr_device, NUM_RAYS * sizeof(Ray));

		cudaMallocManaged(&image_device, NUM_RAYS * 4 * sizeof(uint8_t));	//4 = RGBA
		image_host = new uint8_t[NUM_RAYS * 4];
		printf("RenderEngine initialized. Approx GPU size: %d Mb\n\n", (int)((NUM_RAYS * sizeof(Ray) + vol->len * sizeof(Voxel)) / 1000000.));
	}
	void render(sf::Texture* texture);
	void updateVolume() {
		auto start = chrono::high_resolution_clock::now();
		cudaMemcpy(voxels, volume->voxels, volume->len * sizeof(Voxel), cudaMemcpyHostToDevice);
		printf("Volume moved to GPU in %d ms.\n", chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start));
	}

private:
	Volume* volume;				// Host side
	Voxel* voxels;				// Device side
	Ray* rayptr_device;
	Ray* rayptr_host;
	Camera* camera;
	uint8_t* image_device;
	uint8_t* image_host;

	// Variables
	int blocks_per_sm;
	int stream_size;
	int ray_stream_bytes;
	int image_stream_bytes;


	// Various functions
	void CUDAPlanning() {
		for (int y = 0; y < RAY_BLOCKS_PER_DIM; y++) {
			for (int x = 0; x < RAY_BLOCKS_PER_DIM; x++) {
				//ray_block[y * RAY_BLOCKS_PER_DIM + x] = Float2(x, y);
			}
		}
		blocks_per_sm = NUM_RAYS / (THREADS_PER_BLOCK * N_STREAMS);
		stream_size = blocks_per_sm * THREADS_PER_BLOCK;
		ray_stream_bytes = stream_size * sizeof(Ray);
		image_stream_bytes = stream_size * 4 * sizeof(uint8_t);
		printf("Blocks per SM: %d \n", blocks_per_sm);
	}
	Ray* initRays();



	// Helper functions
	int xyToRayIndex(int x, int y) { return y * RAYS_PER_DIM + x; }
};