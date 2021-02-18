#include "CudaContainers.cuh"

Int3::Int3(CudaFloat3 s) { x = s.x; y = s.y; z = s.z; }




//-------------------------------------------------------------- INITIALIZATION ---------------------------------------------------------------------------------------------\\

void TissueCluster3D::findNeighborsAndMean(Volume* vol) {
	for (int i = 0; i < member_indexes.size(); i++) {
		int member_index = member_indexes[i];
		Voxel* cur_voxel = &vol->voxels[member_index];			// No ignores are added to a cluster
		mean += (double)cur_voxel->hu_val / (double)member_indexes.size();
		Int3 origin = indexToXYZ(member_index, vol->size);
		for (int j = 0; j < 6; j++) {							// search each neighbor-voxel for neighbor cluster
			Int3 pos = getImmediateNeighbor(origin, j);
			if (isInVolume(pos, vol->size)) {
				Voxel* neighbor = &vol->voxels[xyzToIndex(pos, vol->size)];
				if (!neighbor->ignore) {
					if ( neighbor->cluster_id != cur_voxel->cluster_id) {
						neighbor_ids.addVal(neighbor->cluster_id);
					}
				}
			}
		}
	}
}








//-------------------------------------------------------------- MERGING ---------------------------------------------------------------------------------------------\\

void TissueCluster3D::mergeClusters(vector<TissueCluster3D*>* all_clusters) {
	if (dead)
		return;

	verifyNeighborAliveness(all_clusters);

	int num_merges = 0;

	int* ids = neighbor_ids.fetch();
	int num_neighbors = neighbor_ids.size();					// Cache here, as size increases during function!!
	for (int i = 0; i < num_neighbors; i++) {
		int neighbor_id = ids[i];


		TissueCluster3D* neighbor = all_clusters[0][neighbor_id];
		if (neighbor->dead) {
			int parent_id = neighbor->getParentID(all_clusters);
			if (parent_id == -1)								// In case the parent is annihilated
				continue;
			neighbor = all_clusters[0][parent_id];
		}
		if (neighbor->id == id) {
			neighbor_ids.deleteVal(neighbor_id);				// Case only arises from clusters eaten by another cluster eaten by parent :)
			continue;
		}
			
		if (isMergeable(neighbor)) {
			if (member_indexes.size() >= neighbor->member_indexes.size()) {
				num_merges++;
				mergeCluster(all_clusters, neighbor);
			}
			else {
				neighbor->mergeCluster(all_clusters, all_clusters[0][id]);
				neighbor->mergeClusters(all_clusters);		// Continue merging on the new parent, who now owns all of this clusters' neighbors
				delete(ids);
				return;   										// Cannot continue as new parent have deleted everything in this cluster.
			}
		}
		else {
			neighbor_ids.deleteVal(neighbor_id);				// SO YEAH, THIS PROBABLY ISN'T IDEAL, but maybe it's not a problem.
		}
	}
	delete(ids);

	if (num_merges > 0) {
		//mergeClusters(all_clusters);
	}
		
}







void TissueCluster3D::mergeCluster(vector<TissueCluster3D*>* all_clusters, TissueCluster3D* orphan) {	// you know, because it has no parent yet
	mean = (mean * member_indexes.size() + orphan->mean * orphan->member_indexes.size()) / (member_indexes.size() + orphan->member_indexes.size());
	orphan->verifyNeighborAliveness(all_clusters);
	transferMembers(orphan);				// Deletes large structures
	
	neighbor_ids.deleteVal(orphan->id);
	orphan->kill(id);						// Alters orphans ID.
}

inline float TissueCluster3D::mergeCost(TissueCluster3D* orphan) {
	return abs(orphan->mean - mean);
}

bool TissueCluster3D::isMergeable(TissueCluster3D* orphan) {
	int scalar = 1;
	if (member_indexes.size() < 20 || orphan->mean < -100)
		scalar = 3;
	
	if (mergeCost(orphan) < max_difference*scalar) {
		return true;
	}
	return false;
}

void TissueCluster3D::transferMembers(TissueCluster3D* orphan) {
	member_indexes.insert(member_indexes.end(), orphan->member_indexes.begin(), orphan->member_indexes.end());
	orphan->member_indexes.clear();

	int* orphan_neighbors = orphan->neighbor_ids.fetch();
	for (int i = 0; i < orphan->neighbor_ids.size(); i++) {
		if (orphan_neighbors[i] != id) {
			neighbor_ids.addVal(orphan_neighbors[i]);
		}
	}

	orphan->neighbor_ids.clear();
	delete(orphan_neighbors);
}


void TissueCluster3D::kill(int pa) {
	id = pa;
	dead = true;
}
void TissueCluster3D::annihilate(Volume* vol) {
	//printf("Annihilating %d\n", id);
	id = -1;
	dead = true;
	for (int i = 0; i < member_indexes.size(); i++) {
		vol->voxels[member_indexes[i]].ignore = true;
	}
	neighbor_ids.clear();
	member_indexes.clear();
}



//-------------------------------------------------------------- FINALIZATION ---------------------------------------------------------------------------------------------\\

void TissueCluster3D::finalize(Volume* vol, ColorMaker* CM) {
	if (dead)
		return;
	Color temp_color = CM->colorFromHu(mean);  
	//color = CudaColor(temp_color);
	color = color.getRandColor();
	findEdges(vol); 
}



void TissueCluster3D::findEdges(Volume* vol) {
	for (int i = 0; i < member_indexes.size(); i++) {
		int member_index = member_indexes[i];
		Voxel* cur_voxel = &vol->voxels[member_index];
		cur_voxel->cluster_id = id;
		if (isEdge(vol, indexToXYZ(member_index, vol->size), cur_voxel)) {
			edge_member_indexes.push_back(member_index);
			cur_voxel->color = color;
			cur_voxel->isEdge = true;
		}
	}
}
bool TissueCluster3D::isEdge(Volume* vol, Int3 origin, Voxel* v0) {
	for (int i = 0; i < 6; i++) {
		Int3 pos = getImmediateNeighbor(origin, i);
		if (isInVolume(pos, vol->size)) {						// Edges can on purpose not be volume-border voxels
			int neighbor_index = xyzToIndex(pos, vol->size);
			Voxel* neighbor = &vol->voxels[neighbor_index];
			if (neighbor->ignore || neighbor->cluster_id != v0->cluster_id)	// think ignore is implicit in the second comparison, as its always -1
				return true;
		}
	}
	return false;
}







//-------------------------------------------------------------- MERGING - GENERAL ---------------------------------------------------------------------------------------------\\

void TissueCluster3D::verifyNeighborAliveness(vector<TissueCluster3D*>* all_clusters) {
	int* neighbors = neighbor_ids.fetch();
	int num = neighbor_ids.size();
	for (int i = 0; i < num; i++) {
		int n_id = neighbors[i];
		if (all_clusters[0][n_id]->dead) {
			int parent_id = all_clusters[0][n_id]->getParentID(all_clusters);
			if (parent_id != id && parent_id != -1)	// -1 is annihilated case.
				neighbor_ids.addVal(parent_id);
			neighbor_ids.deleteVal(n_id);
		}
	}
	delete(neighbors);
}

void TissueCluster3D::eliminateVesicle(Volume* vol, vector<TissueCluster3D*>* all_clusters, int threshold_size) {
	if (dead || member_indexes.size() > threshold_size)
		return;

	//printf("ID: %d\n", id);
	verifyNeighborAliveness(all_clusters);
	//printf("verified\n");
	int num_merges = 0;

	int* ids = neighbor_ids.fetch();
	int num_neighbors = neighbor_ids.size();
	
	int least_dif = 9999;
	int best_neighbor_index = NULL;

	for (int i = 0; i < num_neighbors; i++) {
		int n_index = ids[i];
		TissueCluster3D* neighbor = all_clusters[0][n_index];
		if (neighbor->dead)													// TEMP FIX!!
			continue;

		int dif = mergeCost(neighbor);
		if (dif < least_dif) {
			least_dif = dif;
			best_neighbor_index = n_index;
		}
	}

	if (best_neighbor_index == NULL)
		annihilate(vol);
	else {
		//printf("lets merge\n");
		all_clusters[0][best_neighbor_index]->mergeCluster(all_clusters, all_clusters[0][id]);
	}
		


}








//-------------------------------------------------------------- BASIC FUNCTIONS ---------------------------------------------------------------------------------------------\\

int TissueCluster3D::getParentID(vector<TissueCluster3D*>* clusters) { // Returns id -1 if clusters is annihinated
	if (dead) {
		if (id == -1)	// Annihilation case
			return -1;
		return clusters[0][id]->getParentID(clusters);
	}
		
	return id;
}

void TissueCluster3D::colorMembers(Volume* vol, CudaColor(c)) {
	for (int i = 0; i < member_indexes.size(); i++) {
		vol->voxels[member_indexes[i]].isEdge = true;
		vol->voxels[member_indexes[i]].color = c;
	}
}