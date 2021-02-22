#pragma once

#include "CudaContainers.cuh"
#include <cuda.h>
#include "cuda_runtime.h"





class LiveEditor {
public:
	LiveEditor(){}
	LiveEditor(Volume* volume) : vol(volume) { num_clusters = vol->num_clusters; makeCompactClusters(); }
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
		cluster_size = vol->compressedclusters[cluster_id].member_indexes.size();
		printf("    Cluster ID: %05d selected. Mean: %f	Size: %f cm3 \n", cluster_id, cluster_mean, cluster_size/1000.);
	}

	Ray* ray;
	short int cluster_id = -1;
	float cluster_mean = 0;
	int cluster_size = 0;
	int wheel_zoom = 0;

private:
	int num_clusters;
	CompactCluster* comclusters;	// Edited here, and fetched often!
	Ray* rayptr_dev;				// Live rayptr


	Volume* vol;					// Contains rendervoxels, and compressed clusterlist (Only has some info from original clusters)


	void resetAlpha();
	void makeCompactClusters() {
		CompactCluster* ComClusters = new CompactCluster[num_clusters];
		CudaColor c;
		int compact_index = 0;
		for (int i = 0; i < num_clusters; i++) {
			ComClusters[i].setAlpha(1.);
			//ComClusters[i].setColor(clusters[i].color);			
			ComClusters[i].setColor(c.getRandColor());
		}

		
		int bytesize = num_clusters * sizeof(CompactCluster);
		printf("Allocating %d KB vram for Compact Clusters\n", bytesize / 1000);
		cudaMallocManaged(&comclusters, bytesize);
		cudaMemcpy(comclusters, ComClusters, bytesize, cudaMemcpyHostToDevice);
		delete(ComClusters);
	}

};