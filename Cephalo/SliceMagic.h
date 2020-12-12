#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


struct Color3 {
	Color3() {};
	Color3(float r, float g, float b) : r(r), g(g), b(b) {}
	Color3(float gray) : r(gray), g(gray), b(gray) {}
	inline Color3 operator*(float s) const { return Color3(r * s, g * s, b * s); }
	float r, g, b;
};

class SliceMagic
{
public:
	SliceMagic();

private:
	const int size = 512;
	const int sizesq = size * size;
	float* original;
	void loadOriginal();
	float* copySlice(float* slice);
	Color3* colorConvert(float* slice);
	void showSlice(Color3* slice, string title);
	void windowSlice(float* slice, float min, float max);
	float median(float* window);
	void medianFilter(float* slice);
	void kMeans(float* slice, int k);
	void assignToMostCommonNeighbor(float* slice, int x, int y);
	void requireMinNeighbors(float* slice, int min);
	//void onMouse(int event, int x, int y, int, void*);
	inline int xyToIndex(int x, int y) { return y * size + x; }
};

