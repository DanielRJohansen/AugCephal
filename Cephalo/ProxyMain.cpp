#include <iostream>
#include "Environment.h"
#include "Raytracing.h"
#include "CudaOps.cuh"
#include <thread>
#include "Containers.h"
#include "SliceMagic.h"
using namespace std;

class A {
public:
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
	
	SliceMagic SliceM;

	Environment Env;
	Env.Run();

	return 0;
}


