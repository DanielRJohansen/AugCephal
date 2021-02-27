#include "LiveEditor.cuh"


void LiveEditor::selectCluster(int pixel_index) {
	ray = &rayptr_dev[pixel_index];
	if (ray->no_hits) {
		cluster_id = -1;
		return;
	}

	short int id = ray->clusterids_hit[0];
	if (id == -1)
		return;


	cluster_id = id;
	wheel_zoom = 0;

	cluster_mean = vol->compressedclusters[cluster_id].mean;
	avg_dens = huToDensSiemens(cluster_mean);

	cluster_size = vol->compressedclusters[cluster_id].member_indexes.size();
	cluster_weight = avg_dens * cluster_size * voxel_volume;


	hidden_mode = false;


	printf("    Cluster ID: %05d selected. Mean HU: %08f\tMembers: %08d\tSize: %f cm3. Weight: %f g \n",
		cluster_id, cluster_mean, cluster_size, cluster_size * voxel_volume * 1. / 1000., cluster_weight);
}

void LiveEditor::isolateCurrentCluster() {
	if (isolated_mode) {
		unisolate();
	}
	else if (cluster_id == -1)
		return;
	else
		isolate();
}
void LiveEditor::hideCurrentCluster() {
	if (cluster_id == -1)
		return;
	else if (hidden_mode) {
		unhide();
	}
	else
		hide();
}

bool LiveEditor::checkRenderFlag() {
	if (render_flag) {
		render_flag = false;
		return true;
	}
	return false;
}
















void LiveEditor::resetClusters() {
	vol->target_cluster = -1;
	vol->boundingbox = vol->og_boundingbox;
	for (int i = 0; i < num_clusters; i++) {
		vol->compactclusters[i].setAlpha(DEFAULT_ALPHA);
	}
	render_flag = true;
	isolated_mode = false;
}

void LiveEditor::isolate() {
	vol->target_cluster = cluster_id;
	vol->boundingbox = vol->compressedclusters[cluster_id].boundingbox;

	render_flag = true;
	isolated_mode = true;
}
void LiveEditor::unisolate() {
	vol->target_cluster = -1;
	vol->boundingbox = vol->og_boundingbox;

	render_flag = true;
	isolated_mode = false;
}

void LiveEditor::hide() {
	vol->compactclusters[cluster_id].setAlpha(0);
	render_flag = true;
	hidden_mode = true;
}
void LiveEditor::unhide() {
	vol->compactclusters[cluster_id].setAlpha(1);
	render_flag = true;
	hidden_mode = false;
}

void LiveEditor::window(int from, int to) {
	for (int i = 0; i < vol->num_clusters; i++) {
		if (vol->compressedclusters[i].mean < from || vol->compressedclusters[i].mean > to) {
			vol->compactclusters[i].setAlpha(0);
		}
		else {
			vol->compactclusters[i].setAlpha(1);
		}
	}
	render_flag = true;
}


CompactCluster* LiveEditor::makeCompactClusters() {
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
	CompactCluster* comclusters;
	cudaMallocManaged(&comclusters, bytesize);
	cudaMemcpy(comclusters, ComClusters, bytesize, cudaMemcpyHostToDevice);
	delete(ComClusters);
	return comclusters;
}

float LiveEditor::huToDensSiemens(float hu) {	// return g/mm3
	float a = -2.1408;
	float b = -0.0004;
	float c = 3.1460;
	float mass_cm3 = a * exp(b * hu) + c;
	return mass_cm3 * 1. / 1000.;
}