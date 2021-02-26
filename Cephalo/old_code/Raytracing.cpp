#include "Raytracing.h"
#include <ctime>

void Raytracer::initRaytracer(Camera *c, sf::Image *im) {
	camera = c;
	image = im;

	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	initRays();



	cout << "Volume size " << VOL_X*VOL_Y*VOL_Z*sizeof(Block)/1000000. << " MB" << endl;
	cout << "Raytracer Initialized" << endl;
}
Raytracer::~Raytracer() {}


void Raytracer::initRays() {
	float rpd = (float)RAYS_PER_DIM;
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			Ray ray;
			float x_ = 0.5 - 0.5 / rpd - x / rpd;// Shift by half increment to have
			float y_ = 0.5 - 0.5 / rpd - y / rpd;
			float d = sqrt(FOCAL_LEN* FOCAL_LEN + x_ * x_ + y_ * y_) ;
			
			ray.rel_unit_vector = Float3(x_, y_, FOCAL_LEN) * (1. / d);	//Make length equal 1		
			rayptr[xyToRayIndex(y, x)] = ray;	// Yes xy is swapped, this works, so schhh!
		}
	}
}

void Raytracer::render(sf::Texture* texture) {
	auto start = chrono::high_resolution_clock::now();
	//castRays();

	CudaOps.rayStepMS(rayptr, CompactCam(camera->origin, camera->plane_pitch, camera->plane_yaw, camera->radius), texture);
	//projectRaysOnPlane();
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	printf("Rendering executed in %d ms.\n", duration);
}





