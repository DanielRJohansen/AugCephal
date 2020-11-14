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
	ColorScheme colorscheme;

private:
	string folder_path = "E:\\NIH_images\\003412_03_01\\";

	void loadScans();
	int xyzToIndex(int x, int y, int z) { return z * 512 * 512 + y * 512 + x; }
	void insertImInVolume(Mat im, int zcoord);

	void medianFilter();
	void cluster();
	void categorizeBlocks();

	void open(int cat_index);
	void close(int cat_index);

	void dilate(int cat_index);
	void erode(int cat_index);
	void updatePreviousCat();

	void assignColorFromCat();


	Block* copyVolume(Block* from);

	
};
