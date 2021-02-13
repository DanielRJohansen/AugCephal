#include "CudaContainers.cuh"

Int3::Int3(CudaFloat3 s) { x = s.x; y = s.y; z = s.z; }






void TissueCluster3D::mergeClusters(Volume* vol, vector<TissueCluster3D*> *all_clusters) {	// you know, because it has no parent yet
	if (dead)
		return;
	//printf("Started merging from cluster %d with %d neighbors\n", id, neighbor_ids.size());
	int* ids = neighbor_ids.fetch();
	for (int i = 0; i < neighbor_ids.size(); i++) {
		int neighbor_id = ids[i];
		if (neighbor_id < 0) {
			printf("CLuster %d has bad neighbor %d. Size: %d: i: %d\n", id, neighbor_id, neighbor_ids.size(), i);
			exit(-1);
		}
		//continue;
			
		//printf("id: %d size: %d\n", neighbor_id, all_clusters->size());
		TissueCluster3D* neighbor = all_clusters[0][neighbor_id];
		if (neighbor->dead)
			continue;
		if (isMergeable(neighbor)) {
			if (member_indexes.size() > neighbor->member_indexes.size()) {
				mergeCluster(vol->voxels, neighbor);
				delete(ids);
				ids = neighbor_ids.fetch();
			}
			else {
				neighbor->mergeCluster(vol->voxels, all_clusters[0][id]);
				neighbor->mergeClusters(vol, all_clusters);		// Continue merging on the new parent, who now owns all of this clusters' neighbors
				delete(ids);
				return;												// Cannot continue as new parent have deleted everything if this cluster.
			}
		}
	}
	delete(ids);
}


void TissueCluster3D::findNeighborsAndMean(Volume* vol) {
	for (int i = 0; i < member_indexes.size(); i++) {
		int member_index = member_indexes[i];
		Voxel* cur_voxel = &vol->voxels[member_index];	// No ignores are added to a cluster
		mean += (double)cur_voxel->hu_val / (double)member_indexes.size();


		Int3 origin = indexToXYZ(member_index, vol->size);

		// search each neighbor-voxel for neighbor cluster
		for (int i = 0; i < 6; i++) {
			Int3 pos = getImmediateNeighbor(origin, i);
			if (isInVolume(pos, vol->size)) {
				Voxel* neighbor = &vol->voxels[xyzToIndex(pos, vol->size)];
				if (!neighbor->ignore && neighbor->cluster_id != cur_voxel->cluster_id) {
					neighbor_ids.addVal(neighbor->cluster_id);
				}
			}
		}
	}
}




void TissueCluster3D::mergeCluster(Voxel* voxels, TissueCluster3D* orphan) {	// you know, because it has no parent yet
	mean = (mean * member_indexes.size() + orphan->mean * orphan->member_indexes.size()) / (member_indexes.size() + orphan->member_indexes.size());

	orphan->reassignMembersClusterID(voxels, id);	// Needed to decide which voxels are edges.

	transferMembers(orphan);				// Deletes large structures
	
	neighbor_ids.deleteVal(orphan->id);
	orphan->kill(id);						// Deletes orphans ID.
}

bool TissueCluster3D::isMergeable(TissueCluster3D* orphan) {
	if (abs(orphan->mean - mean) < max_difference) {
		return true;
	}
	return false;
}

void TissueCluster3D::updateEdges() {

}

void TissueCluster3D::findEdges(Volume* vol) {
	for (int i = 0; i < member_indexes.size(); i++) {
		int member_index = member_indexes[i];
		Voxel* cur_voxel = &vol->voxels[member_index];
		if (isEdge(vol, indexToXYZ(member_index, vol->size), cur_voxel)) {
			edge_member_indexes.push_back(member_index);
			cur_voxel->color = color;
			cur_voxel->isEdge = true;
		}
	}
}

void TissueCluster3D::transferMembers(TissueCluster3D* orphan) {
	member_indexes.insert(member_indexes.end(), orphan->member_indexes.begin(), orphan->member_indexes.end());
	orphan->member_indexes.clear();


	int* orphan_neighbors = orphan->neighbor_ids.fetch();
	for (int i = 0; i < orphan->neighbor_ids.size(); i++)
		neighbor_ids.addVal(orphan_neighbors[i]);
	orphan->neighbor_ids.clear();
	delete(orphan_neighbors);

}
void TissueCluster3D::kill(int pa) {
	id = pa;
	dead = true;
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







