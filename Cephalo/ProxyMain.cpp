#include <iostream>

#include "Environment.h"
#include "Containers.h"
//#include "SliceMagic.h"
#include "CudaContainers.cuh"
#include "Preprocessing.cuh"


// Just for testing
#include "TreeClasses.h"


using namespace std;

class A {
public:
	A() {
		h[0] = 24;
	}
	A(float s) {
		h[0] = s;
	}
	float h[10] = { 0 };
};

void test(int* hllo) {
	hllo = new int[300000];
	hllo[30] = 99;
}
int pointerPLay() {
	int* a = new int[4000];
	int** b = new int* [30];
	for (int i = 0; i < 4000; i++) {
		a[i] = 10;
	}
	for (int i = 0; i < 30; i++) {
		b[i] = &a[i * 10];
	}
	for (int i = 0; i < 30; i++) {
		*b[i] *= 2;
	}
	for (int i = 0; i < 40; i++) {
		printf("%d\n", a[i]);
	}
	return 0;
}


int main() {




	//SliceMagic SliceM;
	Preprocessor PP;
	Volume* vol = PP.processScan("F:\\DumbLesion\\NIH_scans\\002701_04_03\\", Int3(512, 512, 191), 1. / 0.8164);
	Environment Env(vol);
	Env.Run();


	return 0;
}


