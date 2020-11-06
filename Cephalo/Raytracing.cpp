#include "Raytracing.h"
#include <ctime>

void Raytracer::initRaytracer(Camera *c, sf::Image *im, Block* volume) {
	camera = c;
	image = im;

	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	initRays();

	CudaOps.newVolume(volume);

	cout << "Volume size " << VOL_X*VOL_Y*VOL_Z*sizeof(Block)/1000000. << " MB" << endl;
	cout << "Raytracer Initialized" << endl;
}
Raytracer::~Raytracer() {}


void Raytracer::initRays() {
	float rpd = (float)RAYS_PER_DIM;
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			Ray ray;
			float x_ = -0.5 + 0.5 / rpd + x / rpd;// Shift by half increment to have
			float y_ = 0.5 - 0.5 / rpd - y / rpd;
			float d = sqrt(FOCAL_LEN* FOCAL_LEN + x_ * x_ + y_ * y_) ;
			
			ray.rel_unit_vector = Float3(x_, y_, FOCAL_LEN) * (1. / d);	//Make length equal 1
			if (y == 0 && x == 0) {
				ray.rel_unit_vector.print();
				cout << endl;
			}			
			rayptr[xyToRayIndex(y, x)] = ray;	// Yes xy is swapped, this works, so schhh!
		}
	}
}


void Raytracer::render() {
	cout << "rendering" << endl;
	//time_t start, finish;

	//time(&start);
	castRays();
	//time(&finish);
	//float t1 = difftime(finish, start);

	//time(&start);
	CudaOps.rayStep(rayptr);
	//time(&finish);
	//float t2 = difftime(finish, start);

	//time(&start);
	projectRaysOnPlane();
	//time(&finish);
	//float t3 = difftime(finish, start);

	//printf("Cast time: %f   Step time: %f   Projection time: %f", t1, t2, t3);
}




void Raytracer::castRays() {
	for (int i = 0; i < NUM_RAYS; i++) {
		rayptr[i].cam_pitch = camera->plane_pitch;
		rayptr[i].cam_yaw = camera->plane_yaw;
		rayptr[i].origin = camera->origin;
	}
}


void Raytracer::projectRaysOnPlane() {
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float color = rayptr[xyToRayIndex(x, y)].acc_color * 256;
			int c;
			if (color > 255) c = 255;
			else c = (int)color;
			image->setPixel(x, y, sf::Color(c, c, c));
		}
	}
}

