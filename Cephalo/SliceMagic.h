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
	Mask(float custom_mask[25]) {
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
		mean /= total;
		return mean;
	}

	float calcVar(float* kernel, float mean) {
		float var = 0;
		for (int i = 0; i < 25; i++) {
			float dist = kernel[i] - mean;
			var += sqrt(dist * dist) * mask[i];	// Some dist shall weigh more to var
								
		}
		return var / total;
	}
	float total = 0;
	float mask[25] = {0};
	inline int xyC(int x, int y) { return y * 5 + x; }
};


struct Pixel {
	Pixel() {};
	Pixel(float val, int index) : val(val), index(index) {};
	
	Color3 color;
	int index;
	float val = -1;
	int cluster_id = -1;
	int cluster_size = 0;
	float cluster_mean;
	//int cat = -1;
	bool is_edge = false;
	int cluster_changes = 0;


	int neighbor_indexes[9];
	int n_neighbors = 0;

	void addNeighbor(int i) { neighbor_indexes[n_neighbors] = i; n_neighbors++; }
	void checkAssignBelonging(Pixel* image);
	int connectedClusters(Pixel* image, int* connected_indexes);	// The int point should be init to length 9!!
	void assignToCluster(int cl, Color3 co) { cluster_id = cl; color = co; is_edge = false; cluster_changes++; }

	void makeEdge() { is_edge = true; color = Color3(255, 255, 255); }
};

class TissueCluster {
public:
	TissueCluster() {}
	TissueCluster(Pixel p);

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
	Color3 color;
	//float cluster_mean;		// I dont think this is used.

	
private:
	bool deadmarked = false;
	
	int cluster_size = 0;

	float min_val;
	float max_val;

	int allocated_size = 0;
	int* pixel_indexes;
};

class SliceMagic
{
public:
	SliceMagic();

private:

	struct int2 {
		int2() {}
		int2(int x, int y) : x(x), y(y) {}
		int x, y;
	};

	const int size = 512;
	const int sizesq = size * size;
	float* original;
	void loadOriginal();
	float* copySlice(float* slice);
	Color3* colorConvert(float* slice);
	void showSlice(Color3* slice, string title);
	void showImage(Pixel* image, string title);
	inline float normVal(float hu, float min, float max) { return (hu - min) / (max - min); }
	void windowSlice(float* slice, float min, float max);
	
	// Image value only segmentation, obsolete for now
	void kMeans(float* slice, int k, int itereations);
	void assignToMostCommonNeighbor(float* slice, int x, int y);
	void requireMinNeighbors(float* slice, int min);

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
	float min_val = 0.03;
};

