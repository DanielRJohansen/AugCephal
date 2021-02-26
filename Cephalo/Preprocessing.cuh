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
#include <Windows.h>
#include <cstdlib>
#include "CudaContainers.cuh"
#include "GeneralPurposeFunctions.cuh"

#include "resizing.cuh"
#include "FuzzyAssignment.cuh""
#include "ColorMaker.h"
#include "Constants.h"
using namespace std;

typedef vector<string> stringvec;	// Used for reading directory




class Preprocessor {
public:
	Preprocessor() {};


	Volume* processScan(string path, Int3 s, float z_over_xy, float true_voxel_volume) {
		//return new Volume;
		input_size = s;
		raw_scan = new float[s.x*s.y*s.z];
		loadScans(path);
		scan = raw_scan;
		size = s;
		
		//scan = Interpolate3D(raw_scan, size, &size, z_over_xy);		
		Volume* volume = convertToVolume(scan, size);
		volume->true_voxel_volume = true_voxel_volume;
		delete(scan, raw_scan);


		// Algoritmic preprocessing
		windowVolume(volume, HU_MIN, HU_MAX);		// Norm values are set here
		setIgnoreBelow(volume, HU_MIN);
		setColumnIgnores(volume);


		int k = 12;								// Temporarily locked to 12!!!
		rmf(volume);
		removeAirLayer(volume);
		fuzzyClusterAssignment(volume, k, 60);	// Limited to k<=15 for 512 threads pr block.		!! Make intelligent block spread



		// Move voxels to HOST here, get speedup?	
		vector<TissueCluster3D*> clusters = clusterSync(volume);

		int remaining_clusters = mergeClusters(volume, clusters, clusters.size());	
		remaining_clusters = mergeClusters(volume, clusters, remaining_clusters);

		remaining_clusters = eliminateVesicles(volume, clusters, 50, remaining_clusters);	// min size to survive

		remaining_clusters = mergeClusters(volume, clusters, remaining_clusters);
		remaining_clusters = mergeClusters(volume, clusters, remaining_clusters);
		
		finalizeClusters(volume, clusters);
		
		




		volume->compressedclusters = removeExcessClusters(clusters, remaining_clusters);
		volume->rendervoxels = compressVoxels(volume, clusters, remaining_clusters);
		volume->num_clusters = remaining_clusters;
		fitBoundingBoxToVolume(volume);
		printf("\n\nPreprocessing finished!\n\n\n\n");
		return volume;
	}


	Int3 size;
private:
	void speedTest();
	void loadScans(string folder_path);
	void insertImInVolume(cv::Mat img, int z);
	Volume* convertToVolume(float* scan, Int3 size);
	void setIgnoreBelow(Volume* vol, float below);
	//float* makeNormvalCopy(Volume* vol);
	void colorFromNormval(Volume* vol);
	


	void rmf(Volume* vol);
	void removeAirLayer(Volume* volume);
	void fuzzyClusterAssignment(Volume* volume, int k, int max_iterations) {
		FuzzyAssigner FA;
		FA.doFuzzyAssignment(volume, k, max_iterations);
	}



	// Clustering
	TissueCluster3D* clusterAsync(Volume* vol, int* num_clusters, int k);
	vector<TissueCluster3D*> clusterSync(Volume* vol);			// Sets num_clusters
	int mergeClusters(Volume* vol, vector<TissueCluster3D*> clusters, int remaining_clusters);
	int eliminateVesicles(Volume* vol, vector<TissueCluster3D*> clusters, int threshold_size, int remaining_clusters);
	void finalizeClusters(Volume* vol, vector<TissueCluster3D*> clusters);
	int countAliveClusters(vector<TissueCluster3D*> clusters, int from);
	

	TissueCluster3D* Preprocessor::removeExcessClusters(vector<TissueCluster3D*> clusters, int remaining_clusters);
	RenderVoxel* compressVoxels(Volume* vol, vector<TissueCluster3D*> clusters, int remaining_clusters);
	void fitBoundingBoxToVolume(Volume* vol);  // Supposed to recieves the compressed clusterlist!
	
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


