#pragma once

#include <assert.h>
#include <iostream>

class Resizer
{
public:
	Resizer(int size_from, int size_to) : size_from(size_from), size_to(size_to) {};
	float* Interpolate2D(float* slice);

private:
	int size_from, size_to;
	float* from_slice;
	float* to_slice;
	float* getKernel(int x, int y);

	inline int xyToIndex(int x, int y, int size) { return y * size + x; }
	inline bool isLegal(int x, int y, int size) { return (x > 0 && y > 0 && x < size && y < size); };
};

