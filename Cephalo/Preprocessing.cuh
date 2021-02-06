#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <chrono>
#include "math.h"

//Dunno which of these are necessary. Maybe none?
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

// Threading
#include <future>
#include <thread>

#include "CudaContainers.cuh"
#include "GeneralPurposeFunctions.cuh"

#include "resizing.cuh"
#include "FuzzyAssignment.cuh""
using namespace std;

typedef vector<string> stringvec;	// Used for reading directory




class Preprocessor {
public:
	Preprocessor() {};


	Volume* processScan(string path, Int3 s, float z_over_xy) {
		//return new Volume;
		input_size = s;
		raw_scan = new float[s.x*s.y*s.z];
		loadScans(path);
		scan = raw_scan;
		size = s;

		//scan = Interpolate3D(raw_scan, size, &size, z_over_xy);		
		Volume* volume = convertToVolume(scan, size);

		// Algoritmic preprocessing
		windowVolume(volume, -600, 800);		// Norm values are set here
		setIgnoreBelow(volume, -600);
		setColumnIgnores(volume);


		int k = 6;
		rmf(volume);
		fuzzyClusterAssignment(volume, k, 4);	// Limited to k<=15 for 512 threads pr block.		!! Make intelligent block spread

		int num_clusters;
		clusterAsync(volume, &num_clusters, k);
		//clusterSync(volume, &num_clusters);

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
	void fuzzyClusterAssignment(Volume* volume, int k, int max_iterations) {
		FuzzyAssigner FA;
		FA.doFuzzyAssignment(volume, k, max_iterations);
	}

	// Clustering
	TissueCluster3D* clusterAsync(Volume* vol, int* num_clusters, int k);
	TissueCluster3D* clusterSync(Volume* vol, int* num_clusters);

	TissueCluster3D* iniTissueCluster3D(Volume* vol, int num_clusters, Int3 size);
	
	
	
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


