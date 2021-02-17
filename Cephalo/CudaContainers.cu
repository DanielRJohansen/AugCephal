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

void TissueCluster3D::mergeClusters(Volume* vol, vector<TissueCluster3D*>* all_clusters) {	// you know, because it has no parent yet
	if (dead)
		return;

	int num_merges = 0;

	int* ids = neighbor_ids.fetch();
	int num_neighbors = neighbor_ids.size();					// Cache here, as size increases during function!!

	for (int i = 0; i < num_neighbors; i++) {
		int neighbor_id = ids[i];


		TissueCluster3D* neighbor = all_clusters[0][neighbor_id];
		if (neighbor->dead) {
			neighbor = all_clusters[0][neighbor->getParentID(all_clusters)];
		}
		if (neighbor->id == id) {
			neighbor_ids.deleteVal(neighbor_id);				// Should only come from clusters eaten by another cluster eaten by parent :)
			continue;
		}
			
		if (isMergeable(neighbor)) {
			if (member_indexes.size() >= neighbor->member_indexes.size()) {
				num_merges++;
				mergeCluster(vol->voxels, all_clusters, neighbor);
			}
			else {
				neighbor->mergeCluster(vol->voxels, all_clusters, all_clusters[0][id]);
				neighbor->mergeClusters(vol, all_clusters);		// Continue merging on the new parent, who now owns all of this clusters' neighbors
				delete(ids);
				return;   										// Cannot continue as new parent have deleted everything in this cluster.
			}
		}
		else {
			neighbor_ids.deleteVal(neighbor_id);				// SO YEAH, THIS PROBABLY ISN'T IDEAL, but maybe it's not a problem.
		}
	}
	delete(ids);

	/*if (num_merges > 0) {
		printf("%d merges!\n", num_merges);
		mergeClusters(vol, all_clusters);
	}*/
		
}







void TissueCluster3D::mergeCluster(Voxel* voxels, vector<TissueCluster3D*>* all_clusters, TissueCluster3D* orphan) {	// you know, because it has no parent yet
	mean = (mean * member_indexes.size() + orphan->mean * orphan->member_indexes.size()) / (member_indexes.size() + orphan->member_indexes.size());
	orphan->verifyNeighborAliveness(all_clusters);
	transferMembers(orphan);				// Deletes large structures
	
	neighbor_ids.deleteVal(orphan->id);
	orphan->kill(id);						// Alters orphans ID.
}

bool TissueCluster3D::isMergeable(TissueCluster3D* orphan) {
	int scalar = 1;
	if (member_indexes.size() < 20 || orphan->mean < -100)
		scalar = 2;
	
	if (abs(orphan->mean - mean) < max_difference*scalar) {
		return true;
	}
	return false;
}


void TissueCluster3D::transferMembers( TissueCluster3D* orphan) {
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


void TissueCluster3D::verifyNeighborAliveness(vector<TissueCluster3D*>* all_clusters) {
	int* neighbors = neighbor_ids.fetch();
	int num = neighbor_ids.size();
	for (int i = 0; i < num; i++) {
		int id = neighbors[i];
		if (all_clusters[0][id]->dead) {
			int parent_id = all_clusters[0][id]->getParentID(all_clusters);
			if (parent_id != id)
				neighbor_ids.addVal(parent_id);
			neighbor_ids.deleteVal(id);
		}
	}
	delete(neighbors);
}

void TissueCluster3D::kill(int pa) {
	id = pa;
	dead = true;
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







