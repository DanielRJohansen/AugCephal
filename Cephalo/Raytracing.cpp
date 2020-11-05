#include "Raytracing.h"


void Raytracer::initRaytracer(Camera *c, sf::Image *im) {
	camera = c;
	image = im;

	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	initRays();

	blocks = new Block[512 * 512 * 30];
	for (int z = 0; z < 30; z++) {
		for (int y = 0; y < RAYS_PER_DIM; y++) {
			for (int x = 0; x < RAYS_PER_DIM; x++) {
				blocks[z * 512 * 512 + y * 512 + x].color *= z/512.;
			}
		}
	}
	CudaOps.newVolume(blocks);

	cout << "Volume size " << 512*512*30*sizeof(Block)/1000000. << " MB" << endl;
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
	castRays();
	CudaOps.rayStep(rayptr);
	projectRaysOnPlane();	
}




void Raytracer::castRays() {
	for (int i = 0; i < NUM_RAYS; i++) {
		rayptr[i].cam_pitch = camera->plane_pitch;
		rayptr[i].cam_yaw = camera->plane_yaw;
		rayptr[i].origin = camera->origin;
	}
}


void Raytracer::projectRaysOnPlane() {
	for (int y = 0; y < 512; y++) {
		for (int x = 0; x < 512; x++) {
			float color = rayptr[xyToRayIndex(x, y)].acc_color * 256;
			int c;
			if (color > 255) c = 255;
			else c = (int)color;
			image->setPixel(x, y, sf::Color(c, c, c));
		}
	}
}

