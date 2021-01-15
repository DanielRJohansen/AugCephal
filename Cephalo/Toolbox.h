#pragma once

#include <iostream>
#include <vector>
using namespace std;
class Toolbox
{
public:
	float* vectorToList(vector<float> v) {
		int size = v.size();
		float* l = new float[size];
		for (int i = 0; i < size; i++) {
			l[i] = v[i];
		}
		return l;
	}
	float medianOfMedian(float* list, int size){
		int median_index = find_kth(list, size, size / 2);

		return list[median_index];
	}
private:
	int find_kth(float* list, int n, int k, int depth = 0);


};




