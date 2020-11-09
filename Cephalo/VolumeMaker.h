#pragma once
#include <iostream>
#include <string>
#include <windows.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Containers.h"
#include "Constants.h"
#include "CudaOps.cuh"

using namespace std;
using namespace cv;

typedef vector<string> stringvec;


class VolumeMaker
{
public:
	VolumeMaker();
	Block* volume;


private:
	string folder_path = "E:\\NIH_images\\003412_03_01\\";

	void loadScans();
	int xyzToIndex(int x, int y, int z) { return z * 512 * 512 + y * 512 + x; }
	void insertImInVolume(Mat im, int zcoord);
	void medianFilter();
	Block* copyVolume(Block* from);
	void cluster();
	void categorizeBlocks();
};
