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

struct cluster {
	cluster() {};
	cluster(float fraction) {
		centroid = 0.3;
		assigned_val = fraction;
	}
	float assigned_val;
	float centroid;
	float acc_val = 0;
	int num_members = 0;

	void updateCluster() { centroid = acc_val / num_members; num_members = 0; acc_val = 0; };
	void addMember(float member_val) { acc_val += member_val; num_members++; };
	float belonging(float val) {
		float dist = centroid - val;
		return 1 / (dist * dist);
	}
};

struct Mask {
	Mask() {};
	Mask(int x, int y) {
		for (int y_ = y; y_ < y + 3; y_++) {
			for (int x_ = x; x_ < x + 3; x_++) {
				mask[xyC(x_, y_)] = 1;
			}
		}
		num_active = 9;
	};
	Mask(int custom_mask[25], int nu) {
		for (int i = 0; i < 25; i++) {
			mask[i] = custom_mask[i];
		}
		num_active = nu;
	}

	float applyMask(float* kernel) {
		float mean = 0;
		for (int i = 0; i < 25; i++) {
			kernel[i] *= mask[i];
			mean += kernel[i];
		}
		mean /= num_active;
		return mean;
	}

	float calcVar(float* kernel, float mean) {
		float var = 0;
		for (int i = 0; i < 25; i++) {
			float dist = kernel[i] - mean;
			var += dist * dist;
		}
		return var / num_active;
	}
	float num_active = 9;
	float mask[25] = {0};
	inline int xyC(int x, int y) { return y * 5 + x; }
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
	void rotatingMaskFilter(float* slice);
	inline int xyToIndex(int x, int y) { return y * size + x; }
};

