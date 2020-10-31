#include "Raytracing.h"






void Raytracer::initRaytracer(Camera c) { 
	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	camera = c;
	cout << "This far";
	initRays();
	initCuda();	
	volume.testSetup();
	cout << "Raytracer Initialized" << endl;
}
Raytracer::~Raytracer() {}

void Raytracer::initRays() {
	float ray_increment = RAY_RANGE / RAYS_PER_DIM;
	float start_angle = -RAY_RANGE / 2. + ray_increment / 2.;	// Shift by half increment to have
													// same amount of rays above and below 0
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		float relative_pitch = start_angle + ray_increment * y;
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float relative_yaw = start_angle + ray_increment * x;
			Ray ray;
			ray.relative_pitch = relative_pitch;
			ray.relative_yaw = relative_yaw;
			rayptr[xyToRayIndex(x, y)] = ray;
		}
	}
}

void Raytracer::initRenderPlane() {
	for (int i = 0; i < NUM_RAYS; i++) {
		float p0_sub_l0 = 0;
	}
}

void Raytracer::initCuda() {
	all_step_vectors = new float** [RAYS_PER_DIM];
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		all_step_vectors[y] = new float* [RAYS_PER_DIM];
	}
	origin = new float[3];
	origin[0] = camera.x;
	origin[1] = camera.y;
	origin[2] = camera.z;
	CudaOps.update(rayptr);
}


cv::Mat Raytracer::render(Camera c) {
	camera = c;
	cout << "Rendering" << endl;
	precalcSinCos();
	castRays();
	catchRays();
	return projectRaysOnPlane();	//TODO: only calculate rays belonging to pixels,
									// if focal point has changed.
}

void Raytracer::updateCameraOrigin() {
	origin[0] = camera.x;
	origin[1] = camera.y;
	origin[2] = camera.z;
}

void Raytracer::precalcSinCos() {
	for (int i = 0; i < RAYS_PER_DIM; i++) {
		sin_pitches[i] = sin(rayptr[xyToRayIndex(0, i)].relative_pitch + camera.plane_pitch);
		cos_pitches[i] = cos(rayptr[xyToRayIndex(0, i)].relative_pitch + camera.plane_pitch);
		sin_yaws[i] = sin(rayptr[xyToRayIndex(i, 0)].relative_yaw + camera.plane_yaw);
		cos_yaws[i] = cos(rayptr[xyToRayIndex(i, 0)].relative_yaw + camera.plane_yaw);
	}
}

void Raytracer::castRays() {
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float x_ = 1 * sin_pitches[y] * cos_yaws[x];
			float y_ = 1 * sin_pitches[y] * sin_yaws[x];
			float z_ = 1 * cos_pitches[y];
			
			rayptr[xyToRayIndex(x, y)].step_vector[0] = x_ * RAY_STEPSIZE;
			rayptr[xyToRayIndex(x, y)].step_vector[1] = y_ * RAY_STEPSIZE;
			rayptr[xyToRayIndex(x, y)].step_vector[2] = z_ * RAY_STEPSIZE;

		}
	}
	CudaOps.update(rayptr);
	CudaOps.rayStep();
	cout << "Rays cast" << endl;

}

void Raytracer::catchRays() {
	for (int i = 200; i < 500; i++) {
		for (int j = 0; j < NUM_RAYS; j++) {
			//rayptr[j].rayStep(i);
		}
	}
}

cv::Mat Raytracer::projectRaysOnPlane() {
	cv::Mat image;
	return image;
}

