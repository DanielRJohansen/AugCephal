#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>

#include "CudaContainers.cuh"
#include "Resizer.h"
#include "Containers.h"	// Temporary!!!!
using namespace std;

typedef vector<string> stringvec;	// Used for reading directory


class Preprocessor {
public:
	Preprocessor() {};


	Volume* processScan(string path, Int3 s, float z_over_xy) {
		input_size = s;
		loadScans(path);
		size = resizer.Interpolate3D(raw_scan, resized_scan, input_size.x, 1024, input_size.z, z_over_xy);
		len = size.x * size.y * size.z;
		Volume* volume = convertToVolume(resized_scan, size);


		delete(raw_scan, resized_scan);
		return volume;
	}
	Block* VolToBlockvol(Volume* vol, Int3 size) {
		int s = (size.x * size.y * size.z);
		Block* vol_ = new Block[s];
		for (int i = 0; i < s; i++) {
			vol_[i].cat = 0;
			float c = (float) (vol->voxels[i].norm_val*255);	// floats wtf
			vol_[i].color = Color(c, c, c);
		}
	}


	Int3 size;
	int len;
private:
	void loadScans(string folder_path);
	void insertImInVolume(cv::Mat img, int z);
	Volume* convertToVolume(float* scan, Int3 size) {
		Volume* vol = new Volume(size, resized_scan);
	}


	void windowVolume(Volume* vol, int min, int max) {
		for (int i = 0; i < vol->len; i++) {
			int hu = vol->voxels[i].hu_val - 32768;
			if (hu > max) { vol->voxels[i].norm_val = 1; }
			else if (hu < min) { vol->voxels[i].norm_val = 0; }
			else vol->voxels[i].norm_val = (hu - min) / (max - min);		
		}
	}



	inline int xyzToIndex(Int3 coord, Int3 size) {
		return coord.z * size.y * size.x + coord.y * size.x + coord.x;
	}


	Int3 input_size;
	float* raw_scan;
	float* resized_scan;


	Resizer resizer;












	// Hopefully temporary
	void read_directory(const string& name, stringvec& v)
	{
		string pattern(name);
		pattern.append("\\*");
		WIN32_FIND_DATA data;
		HANDLE hFind;
		if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
			do {
				v.push_back(data.cFileName);
			} while (FindNextFile(hFind, &data) != 0);
			FindClose(hFind);
		}
	}
};


