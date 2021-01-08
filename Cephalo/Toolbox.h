#pragma once

#include <iostream>
using namespace std;
class Toolbox
{
public:
	float medianOfMedian(float* list, int size){
		int median_index = find_kth(list, size, size / 2);

		return list[median_index];
	}
private:
	int find_kth(float* list, int n, int k, int depth = 0);


};

