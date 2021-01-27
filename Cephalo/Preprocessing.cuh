#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <chrono>

#include "CudaContainers.cuh"
//#include "CudaOps.cuh"
using namespace std;

typedef vector<string> stringvec;	// Used for reading directory


float* Interpolate3D(float* raw_scan, Int3 size_from, Int3* size_to, float z_over_xy);	//Final arg refers to pixel spacing. Returns new size.


class Preprocessor {
public:
	Preprocessor() {};


	Volume* processScan(string path, Int3 s, float z_over_xy) {
		speedTest();
		return new Volume;
		input_size = s;
		len = size.x * size.y * size.z;
		raw_scan = new float[len];
		loadScans(path);

		resized_scan = Interpolate3D(raw_scan, input_size, &size, z_over_xy);		
		delete(raw_scan);
		Volume* volume = convertToVolume(resized_scan, size);

		windowVolume(volume, -700, 800);
		setIgnoreBelow(volume, -600);



		printf("Preprocessing finished!\n\n");
		return volume;
	}


	Int3 size;
private:
	void speedTest();
	void loadScans(string folder_path);
	void insertImInVolume(cv::Mat img, int z);
	Volume* convertToVolume(float* scan, Int3 size);
	void setIgnoreBelow(Volume* vol, float below) {
		for (int i = 0; i < vol->len; i++) {
			if (vol->voxels[i].hu_val < below)
				vol->voxels[i].ignore = true;
		}
	}


	void windowVolume(Volume* vol, int min, int max) {
		auto start = chrono::high_resolution_clock::now();
		for (int i = 0; i < vol->len; i++) 	
			vol->voxels[i].norm(min, max);
		
		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
		printf("Volume windowed in %d ms.\n", duration);
	}
	


	


	Int3 input_size;
	int len;
	float* raw_scan;
	float* resized_scan;













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


