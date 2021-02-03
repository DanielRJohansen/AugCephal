#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <chrono>

//Dunno which of these are necessary. Maybe none?
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "CudaContainers.cuh"
//#include "CudaOps.cuh"
using namespace std;

typedef vector<string> stringvec;	// Used for reading directory


float* Interpolate3D(float* raw_scan, Int3 size_from, Int3* size_to, float z_over_xy);	//Final arg refers to pixel spacing. Returns new size.


class Preprocessor {
public:
	Preprocessor() {};


	Volume* processScan(string path, Int3 s, float z_over_xy) {
		input_size = s;
		raw_scan = new float[s.x*s.y*s.z];
		loadScans(path);
		scan = raw_scan;
		size = s;

		//scan = Interpolate3D(raw_scan, size, &size, z_over_xy);		
		Volume* volume = convertToVolume(scan, size);

		// Algoritmic preprocessing
		windowVolume(volume, -700, 800);		// Norm values are set here
		setIgnoreBelow(volume, -600);
		setColumnIgnores(volume);



		//rmf(volume);
		kMeans(volume, 12);
		colorFromNormval(volume);


		printf("Preprocessing finished!\n\n");
		return volume;
	}


	Int3 size;
private:
	void speedTest();
	void loadScans(string folder_path);
	void insertImInVolume(cv::Mat img, int z);
	Volume* convertToVolume(float* scan, Int3 size);
	void setIgnoreBelow(Volume* vol, float below);
	float* makeNormvalCopy(Volume* vol);
	void colorFromNormval(Volume* vol);



	void rmf(Volume* vol);
	CudaKCluster* kMeans(Volume* volume, int k);

	void setColumnIgnores(Volume* vol);



	// Algorithmic preprocessing
	void windowVolume(Volume* volume,  float min, float max);



	

	Int3 input_size;
	float* raw_scan;
	float* scan;













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


