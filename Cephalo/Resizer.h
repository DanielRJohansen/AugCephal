#pragma once

#include <assert.h>
#include <iostream>

class Resizer
{
public:
	Resizer(int size_from, int size_to) : size_from(size_from), size_to(size_to) {};
	float* Interpolate2D(float* slice);

	// INTERPOLATION CAN ONLY HANDLE DOUBLING THE XY SIZE!
	float* Interpolate3D(float* slices, int xy_from, int xy_to, int num_slices, float z_over_xy);	//Final arg refers to pixel spacing

private:
	int size_from, size_to;
	float* from_slice;
	float* to_slice;


	int width_old, width_new, height_old, height_new;

	float* getKernel(int x, int y);
	void getKernel(float* kernel, int x, int y, int z);
	inline int xyToIndex(int x, int y, int size) { return y * size + x; }
	inline bool isLegal(int x, int y, int size) { return (x > 0 && y > 0 && x < size && y < size); };

	inline bool isLegal(int x, int y, int z, int width, int height) { 
		return (x >= 0 && y >= 0 && z >= 0 && x < width && y < width && z < height); // Why was this > not >= before?
	};	
	inline int xyzToIndex(int x, int y, int z, int width, int height) { return z * width * width + y * width + x; }

};

