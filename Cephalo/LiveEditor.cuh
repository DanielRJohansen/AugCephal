#pragma once

#include "CudaContainers.cuh"
#include <cuda.h>
#include "cuda_runtime.h"
#include <math.h>





class LiveEditor {
public:
	LiveEditor(){}
	LiveEditor(Volume* volume) : vol(volume) { 
		voxel_volume = volume->true_voxel_volume;
		num_clusters = vol->num_clusters; 
	}
	void setRayptr(Ray* rayptr) { rayptr_dev = rayptr; }




	void selectCluster(int pixel_index);
	RenderCluster* makeRenderClusters();


	void isolateCurrentCluster();
	void hideCurrentCluster();
	void window(int from, int to);
	void resetClusters();



	bool checkRenderFlag();




	Ray* ray;
	int cluster_id = -1;
	float cluster_mean = 0;
	float avg_dens = 0;
	int cluster_members = 0;
	float cluster_weight = 0;
	int wheel_zoom = 0;

private:
	int num_clusters;
		// Edited here, and fetched often!
	Ray* rayptr_dev;				// Live rayptr
	float voxel_volume;				// in mm3

	Volume* vol;					// Contains rendervoxels, and compressed clusterlist (Only has some info from original clusters)

	bool isolated_mode = false;
	bool hidden_mode = false;

	bool render_flag = false;


	void isolate();
	void unisolate();
	void hide();
	void unhide();

	float huToDensSiemens(float hu);	// return g/mm3

};