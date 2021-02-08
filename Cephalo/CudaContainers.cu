#include "CudaContainers.cuh"

Int3::Int3(CudaFloat3 s) { x = s.x; y = s.y; z = s.z; }





unsigned int TissueCluster3D::determineEdges(Volume* vol) {
	for (int i = 0; i < n_members; i++) {
		Voxel voxel = vol->voxels[i];
		mean += voxel.hu_val / n_members;
		int member_index = member_indexes[i];

		Int3 origin = indexToXYZ(member_index, vol->size);

		if (isEdge(vol, origin, voxel)) {				// Also adds potential neighbors to list
			edge_member_indexes.addVal(member_index);
			n_edge_members++;
			vol->voxels[member_index].isEdge = true;
		}
	}
	n_neighbors = neighbor_ids.size();
	return n_members - n_edge_members;
}




void TissueCluster3D::mergeClusters(Voxel* voxels, TissueCluster3D* orphan) {	// you know, because it has no parent yet
	int* ids = neighbor_ids.fetch();
	for (int i = 0; i < n_neighbors; i++) {
		TissueCluster3D* neighbor = &orphan[ids[i]];
		if (isMergeable(neighbor)) {
			mergeCluster(voxels, neighbor);
			neighbor_ids.deleteVal(ids[i]);	// Remove from neighbor list
		}
	}
	delete(ids);
	refactorEdges();
}


void TissueCluster3D::mergeCluster(Voxel* voxels, TissueCluster3D* orphan) {	// you know, because it has no parent yet
	mean = (mean * n_members + orphan->mean * orphan->n_members) / (n_members + orphan->n_members);
	n_members += orphan->n_members;

	transferMembers(orphan);
	
	n_neighbors = neighbor_ids.size();

	orphan->reassignMembers(voxels, id);
}

bool TissueCluster3D::isMergeable(TissueCluster3D* orphan) {
	if (abs(orphan->mean - mean) < max_difference) {

	}
}

void TissueCluster3D::transferMembers(TissueCluster3D* orphan) {
	member_indexes.insert(member_indexes.end(), orphan->member_indexes.begin(), orphan->member_indexes.end());
	orphan->member_indexes.clear();

	int* orphan_edges = orphan->edge_member_indexes.fetch();
	for (int i = 0; i < orphan->edge_member_indexes.size(); i++)
		edge_member_indexes.addVal(orphan_edges[i]);
	orphan->edge_member_indexes.clear();
	delete(orphan_edges);
	n_edge_members = edge_member_indexes.size();

	int* orphan_neighbors = orphan->neighbor_ids.fetch();
	for (int i = 0; i < orphan->neighbor_ids.size(); i++)
		neighbor_ids.addVal(orphan_neighbors[i]);
	orphan->neighbor_ids.clear();
	delete(orphan_neighbors);
}
bool TissueCluster3D::isEdge(Volume* vol, Int3 origin, Voxel v0) {
	bool is_edge = false;

	for (int i = 0; i < 6; i++) {
		Int3 pos = getImmediateNeighbor(origin, i);
		if (isInVolume(pos, vol->size)) {						// Edges can on purpose not be volume-border voxels
			int neighbor_index = xyzToIndex(pos, vol->size);
			Voxel* neighbor = &vol->voxels[neighbor_index];
			if (neighbor->ignore)
				is_edge = true;
			else if (neighbor->cluster_id != v0.cluster_id) {
				is_edge = true;
				addNeighbor(neighbor->cluster_id);
			}
			//addNeighbor(neighbor->cluster_id);
		}
	}
	return is_edge;
}