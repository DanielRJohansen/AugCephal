#pragma once

#include "CudaContainers.cuh"
#include <cuda.h>
#include "cuda_runtime.h"





class LiveEditor {
public:
	LiveEditor(){}
	LiveEditor(TissueCluster3D* compressed_clusters, int num_clusters) : num_clusters(num_clusters), clusters(compressed_clusters) { makeCompactClusters(); }

	CompactCluster* getCompactClusters() {
		return comclusters;
	}
	CompactCluster* window(int from, int to);



private:
	int num_clusters;
	TissueCluster3D* clusters;
	CompactCluster* comclusters;



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