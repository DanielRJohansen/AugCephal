#include <iostream>

#include "Environment.h"


// Just for testing
#include "TreeClasses.h"
#include "Testing\Suite.h"

using namespace std;


int main() {
	//Suite s;
	//return 1;


	float voxel_volume = 0.816406 * 0.816406 * 0.816406;
	Environment Env("F:\\DumbLesion\\NIH_scans\\002701_04_03\\", Int3(512, 512, 191), 1. / 0.8164, voxel_volume);
	Env.Run();

	return 0;
}


