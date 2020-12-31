#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdlib>


using namespace std;
using namespace cv;





struct Color3 {
	Color3() {};
	Color3(float r, float g, float b) : r(r), g(g), b(b) {}
	Color3(float gray) : r(gray), g(gray), b(gray) {}
	Color3 getRandColor() { return Color3(rand() % 255, rand() % 255, rand() % 255); }
	inline Color3 operator*(float s) const { return Color3(r * s, g * s, b * s); }
	float r, g, b;
};



struct Kcluster {
	Kcluster() {};
	Kcluster(float fraction) {
		centroid = 0.3;
		assigned_val = fraction;
	}
	float assigned_val;
	float centroid;
	float acc_val = 0;
	int num_members = 0;

	void updateCluster() { centroid = acc_val / num_members; num_members = 0; acc_val = 0; };
	void addMember(float member_val) { acc_val += member_val; num_members++; };
	float belonging(float val) {
		float dist = centroid - val;
		return 1 / (dist * dist);
	}
};


float medianOfList(float* list, int size);

struct Mask {
	Mask() {};
	Mask(int x, int y) {
		for (int y_ = y; y_ < y + 3; y_++) {
			for (int x_ = x; x_ < x + 3; x_++) {
				mask[xyC(x_, y_)] = 1;
				total += mask[xyC(x_, y_)];
			}
		}
	};
	Mask(int custom_mask[25]) {
		for (int i = 0; i < 25; i++) {
			mask[i] = custom_mask[i];
			total += mask[i];
		}
		
	}

	float applyMask(float* kernel) {
		float mean = 0;
		for (int i = 0; i < 25; i++) {
			kernel[i] *= mask[i];
			mean += kernel[i];
		}
		mean /= (float) total;
		return mean;
	}

	float calcVar(float* kernel, float mean) {
		float var = 0;
		for (int i = 0; i < 25; i++) {
			float dist = kernel[i] - mean;
			var += sqrt(dist * dist) * mask[i];	// Some dist shall weigh more to var
								
		}
		return var / (float) total;
	}

	float median(float* kernel) {
		float* list = new float[total];
		int head = 0;
		for (int i = 0; i < 25; i++) {
			for (int j = 0; j < mask[i]; j++) {
				list[head] = kernel[i];
				head++;
			}
		}
		return medianOfList(list, total);
	}

	int total = 0;
	int mask[25] = {0};
	inline int xyC(int x, int y) { return y * 5 + x; }
};


class Pixel {
public:
	Pixel() {};
	Pixel(float val, int index) : val(val), index(index) {};

	int index;
	int cluster_id = -1;	// NOT ASSIGNED
	Color3 color;

	void addNeighbor(int i) { neighbor_indexes[n_neighbors] = i; n_neighbors++; }
	void checkAssignBelonging(Pixel* image);	// ILLEGAL, DOES NOT UPDATE CLUSTER MEMBER COUNT
	int connectedClusters(Pixel* image, int* connected_indexes);	// The int point should be init to length 9!!
	void assignToCluster(int cl, Color3 co, float mean) { cluster_id = cl; color = co; is_edge = false; cluster_changes++; cluster_mean = mean; }
	inline void reserve() { reserved = true; }
	inline bool isReserved() { return reserved; }
	inline bool isEdge() { return is_edge; }
	inline void makeEdge() { is_edge = true; color = Color3(255, 255, 255); }
	inline float getVal() { return val; }
	inline int getID() { return cluster_id; }
	inline float getClusterMean() { return cluster_mean; }

	void assignToBestNeighbor(Pixel* image) {
		if (!is_edge) return;
		int best_index = -1;
		float min_dist = 999999;
		for (int i = 0; i < n_neighbors; i++) {
			int index = neighbor_indexes[i];
			Pixel p = image[index];
			if (p.isEdge())
				continue;
			float dist = sqrt((p.cluster_mean - cluster_mean) * (p.cluster_mean - cluster_mean));
			if (dist < min_dist) {
				min_dist = dist;
				best_index = index;
			}
		}
		if (best_index == -1) {
			printf("What the fuck, no neighbors?");
			return;
		}
		Pixel p = image[best_index];
		assignToCluster(p.cluster_id, p.color, p.getClusterMean());
	}

private:
	bool reserved = false;	// used for initial clustering ONLY

	float val;
	int cluster_size = 0;
	float cluster_mean = 0;
	//int cat = -1;
	bool is_edge = false;
	int cluster_changes = 0;


	int neighbor_indexes[9];
	int n_neighbors = 0;

};

class TissueCluster {
public:
	TissueCluster() {}

	bool isMergeable(TissueCluster** clusters, int num_clusters, float absolute_dif, float relative_dif);
	void mergeClusters(TissueCluster** clusters, Pixel* image, int num_clusters);	// remember to set min and max here
	void addToCluster(Pixel p);
	void handleArraySize();
	void deadmark() { deadmarked = true; delete(pixel_indexes); }

	//inline float getMean() { return cluster_mean; }
	//printf("Here is may go wrong: %d\n", cluster_id);
	inline float getMin() {  return min_val; }
	inline float getMax() { return max_val; }
	inline int getPixel(int index) { return pixel_indexes[index]; }
	inline int getSize() { return cluster_size; }
	bool isDeadmarked() { return deadmarked; }

	int cluster_id;
	bool initialized = false;
	Color3 color = Color3().getRandColor();
	float cluster_mean = 0;		// I dont think this is used.

	
private:
	bool deadmarked = false;
	
	int cluster_size = 0;

	float min_val;
	float max_val;

	// Member pixels
	int allocated_size = 0;
	int* pixel_indexes;

	// Neighbors
	int num_neighbors = 0;
	int* neighbor_ids;
};

//const int size = 512;
const string ip = "D:\\DumbLesion\\NIH_scans\\Images_png\\002701_04_03\\160.png";
const int ss = 1024;
//const string ip = "E:\\NIH_images\\000330_06_01\\183.png";
class SliceMagic
{
public:
	SliceMagic();
	inline int getSize() { return size; }
private:

	struct int2 {
		int2() {}
		int2(int x, int y) : x(x), y(y) {}
		int x, y;
	};


	const int size = ss;
	const int load_size = 512;
	const string im_path = ip;
	const int sizesq = size * size;
	//float* original;



	float* loadOriginal();
	float* copySlice(float* slice);
	Color3* colorConvert(float* slice);
	void showSlice(Color3* slice, string title, int s=-1);
	void showImage(Pixel* image, string title);
	inline float normVal(float hu, float min, float max) { return (hu - min) / (max - min); }
	void windowSlice(float* slice, float min, float max);
	
	inline bool isLegal(int x, int y) { return x >= 0 && y >= 0 && x < size && y < size; }

	// Image value only segmentation, obsolete for now
	void kMeans(float* slice, int k, int itereations);
	void assignToMostCommonNeighbor(float* slice, int x, int y);
	void requireMinNeighbors(float* slice, int min);

	void histogramFuckingAround(float* slice);

	float median(float* window);
	void medianFilter(float* slice);
	void rotatingMaskFilter(float* slice, int num_masks);

	void hist(float* slice);
	void deNoiser(float* slice);

	void propagateHysteresis(float* slice, bool* forced_black, float* G, int x, int y);
	void hysteresis(float* slice, float* G, bool* forced_black);



	void sliceToImage(float* slice, Pixel* image) { 
		for (int i = 0; i < sizesq; i++) {
			image[i] = Pixel(slice[i], i);
			//printf("%f\n", image[i].getVal());
			//printf("%f\n", slice[i]);
		}
		findNeighbors(image);
	}
	void applyEdges(float* slice, Pixel* image) { for (int i = 0; i < sizesq; i++) if (slice[i] == 1) image[i].makeEdge(); }
	void propagateCluster(Pixel* image, int cluster_id, Color3 color, float* acc_mean, int* n_members, int* member_indexes, int2 pos);
	int cluster(Pixel* image);	// Returns num clusters

	void mergeClusters(Pixel* image, int num_clusters, float max_absolute_dist, float max_fractional_dist);



	void findNeighbors(Pixel* image);

	void nonMSStep(vector<int>* edge_i, float* s_edge, int* s_index, float* G, float* t, bool* fb, int inc, 
		int x, int y, int y_step, int x_step, int step_index, float threshold);
	void nonMS(float* slice, float* G, float* theta, bool* forced_black, float threshold);
	void applyCanny(float* slice);
	inline int xyToIndex(int x, int y) { return y * size + x; }
	inline int xyToIndex(int2 pos) { return pos.y * size + pos.x; }



	float grad_threshold = 0.08;// 0.12;
	float min_val = 0.035;
};

