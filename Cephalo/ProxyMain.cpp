#include <iostream>

#include "Environment.h"


// Just for testing
#include "TreeClasses.h"


using namespace std;

class A {
public:
	A() { vec.push_back(0.001); }
	A(float s) {
		b = s;
		vec.push_back(s);
		print();
	}
	float b=10;
	vector<float> vec;


	void print() {
		//printf("");
		printf("printing: %f\n", b);
	}
	float sass() { return b; }
};

void test(vector<A*> ref) {
	//for (A* a = *ref->begin(); a < *ref->end(); a+=1) {
	for (int i = 0; i < ref.size(); i++) {
		A* a = ref.at(i);
		a->print();
	}

}

void test(A* arr, int size) {
	for (int i = 0; i < size; i++)
		printf("%f\n", arr[i].vec[0]);
}


vector<A*> one() {
	vector<A*> obj;
	A* elem1 = new A(0.1);
	obj.push_back(elem1);
	A* elem2 = new A(0.2);
	obj.push_back(elem2);
	
	return obj;
}
vector<A*> *two() {
	vector<A*> *obj = new vector<A*>;
	obj->push_back(&A(0.2));
	obj->push_back(&A(0.3));
	return obj;
}
A* tre() {
	A* arr = new A[10];
	arr[0] = A(111);
	arr[1] = A(222);
	for (int i = 2; i < 10; i++)
		arr[i] = A();
	return arr;
}

int main() {
	Environment Env("F:\\DumbLesion\\NIH_scans\\002701_04_03\\", Int3(512, 512, 191), 1. / 0.8164);
	Env.Run();

	return 0;
}


