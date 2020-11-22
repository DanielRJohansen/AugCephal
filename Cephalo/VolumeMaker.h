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
	VolumeMaker(bool default_config);

	bool setIgnore(int cat_index, bool hide);

	Block* volume;
	ColorScheme colorscheme;

private:
	//string folder_path = "E:\\NIH_images\\003412_03_01\\";
	string folder_path = "E:\\NIH_images\\002411_02_02\\";
	//Initial functions
	void loadScans();
	void insertImInVolume(Mat im, int zcoord);

	// Helper functions
	int xyzToIndex(int x, int y, int z);
	bool isLegal(int x, int y, int z);
	bool isNotClustered(int bi);
	bool isCategory(int bi, int cat_id);

	void medianFilter();
	void categorizeBlocks();

	void cluster(int category_index, int min_cluster_size);
	int propagateCluster(int x, int y, int z, int cluster_id, int category_index, int depth);
	//returns the increase in cluster size. Requires reserve stack size 4Gb.

	void open(int cat_index);
	void close(int cat_index);
	void dilate(int cat_index);
	void erode(int cat_index);
	void updatePreviousCat();

	bool ignores[6] = { false };

	void setIgnores(vector<int> ignores);

	void assignColorFromCat();

	Block* copyVolume(Block* from);
	
};
