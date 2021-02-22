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
		makeCompactClusters(); 
	}
	void setRayptr(Ray* rayptr) { rayptr_dev = rayptr; }


	CompactCluster* getCompactClusters() {
		return comclusters;
	}
	CompactCluster* window(int from, int to);

	void selectCluster(int pixel_index) {
		ray = &rayptr_dev[pixel_index];
		short int id = ray->clusterids_hit[0];
		if (id == -1)
			return;

		
		cluster_id = id;
		wheel_zoom = 0;

		cluster_mean = vol->compressedclusters[cluster_id].mean;
		avg_dens = huToDensSiemens(cluster_mean);

		cluster_size = vol->compressedclusters[cluster_id].member_indexes.size();
		cluster_weight = avg_dens * cluster_size * voxel_volume;

		printf("    Cluster ID: %05d selected. Avg. density: %f g/cm3 	Size: %f cm3. Weight: %f g \n", 
			cluster_id, avg_dens*1000, cluster_size * voxel_volume * 1./1000., cluster_weight);
	}

	void isolateCurrentCluster() {
		if (isolated_mode) {
			resetAlpha();
		}
		else if (cluster_id == -1)
			return;
		else
			isolate();
	}

	bool checkRenderFlag() {
		if (render_flag) {
			render_flag = false;
			return true;
		}
		return false;
	}




	Ray* ray;
	short int cluster_id = -1;
	float cluster_mean = 0;
	float avg_dens = 0;
	int cluster_size = 0;
	float cluster_weight = 0;
	int wheel_zoom = 0;

private:
	int num_clusters;
	CompactCluster* comclusters;	// Edited here, and fetched often!
	Ray* rayptr_dev;				// Live rayptr
	float voxel_volume;				// in mm3

	Volume* vol;					// Contains rendervoxels, and compressed clusterlist (Only has some info from original clusters)

	bool isolated_mode = false;

	bool render_flag = false;

	void resetAlpha() {
		for (int i = 0; i < num_clusters; i++) {
			vol->compactclusters[i].setAlpha(DEFAULT_ALPHA);
		}
		render_flag = true;
		isolated_mode = false;
	}

	void isolate() {
		for (int i = 0; i < num_clusters; i++) {
			if (i == cluster_id)
				vol->compactclusters[i].setAlpha(1.);
			else
				vol->compactclusters[i].setAlpha(0.);
		}
		render_flag = true;
		isolated_mode = true;
	}
	void makeCompactClusters() {
		CompactCluster* ComClusters = new CompactCluster[num_clusters];
		CudaColor c;
		int compact_index = 0;
		for (int i = 0; i < num_clusters; i++) {
			ComClusters[i].setAlpha(DEFAULT_ALPHA);
			//ComClusters[i].setColor(clusters[i].color);			
			ComClusters[i].setColor(c.getRandColor());
		}

		
		int bytesize = num_clusters * sizeof(CompactCluster);
		printf("Allocating %d KB vram for Compact Clusters\n", bytesize / 1000);
		cudaMallocManaged(&comclusters, bytesize);
		cudaMemcpy(comclusters, ComClusters, bytesize, cudaMemcpyHostToDevice);
		delete(ComClusters);
	}

	float huToDensSiemens(float hu) {	// return g/mm3
		float a = -2.1408;
		float b = -0.0004;
		float c = 3.1460;
		float mass_cm3 = a * exp(b * hu) + c;
		return mass_cm3 * 1. / 1000.;
	}
};