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
		input_size = s;
		raw_scan = new float[s.x*s.y*s.z];
		loadScans(path);
		scan = raw_scan;
		size = s;

		scan = Interpolate3D(raw_scan, size, &size, z_over_xy);		
		Volume* volume = convertToVolume(scan, size);

		// Algoritmic preprocessing
		windowVolume(volume, -700, 800);




		setIgnoreBelow(volume, -600);
		setColumnIgnores(volume);


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


