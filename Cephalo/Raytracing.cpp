#include "Raytracing.h"

Ray::Ray(float rel_pitch, float rel_yaw) {
	relative_pitch = rel_pitch;
	relative_yaw = rel_yaw;	
}



float* Ray::makeStepVector(RayInfo RF) {
	float x = radius * RF.sin_pitch * RF.cos_yaw;
	float y = radius * RF.sin_pitch * RF.sin_yaw;
	float z = radius * RF.cos_pitch;
	step_vector[0] = x * RAY_STEPSIZE;
	step_vector[1] = y * RAY_STEPSIZE;
	step_vector[2] = z * RAY_STEPSIZE;

	return step_vector;
}







void Raytracer::initRaytracer(Camera c) { 
	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	camera = c;
	initRays();
	initCuda();
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
			rayptr[xyToIndex(x, y)] = Ray(relative_pitch, relative_yaw);
		}
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
	CudaOps.update(all_step_vectors, origin);
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
		sin_pitches[i] = sin(rayptr[xyToIndex(0, i)].relative_pitch + camera.plane_pitch);
		cos_pitches[i] = cos(rayptr[xyToIndex(0, i)].relative_pitch + camera.plane_pitch);
		sin_yaws[i] = sin(rayptr[xyToIndex(i, 0)].relative_yaw + camera.plane_yaw);
		cos_yaws[i] = cos(rayptr[xyToIndex(i, 0)].relative_yaw + camera.plane_yaw);
	}
}

void Raytracer::castRays() {
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			all_step_vectors[y][x] = rayptr[xyToIndex(x, y)].makeStepVector(
				RayInfo(sin_pitches[y], cos_pitches[y], sin_yaws[x], cos_yaws[x]));
		}
	}
	CudaOps.update(all_step_vectors, origin);
	cout << "Rays cast" << endl;
	cout << CudaOps.all_step_vectors[120][121][0] << endl;
	cout << CudaOps.all_step_vectors[120][121][1] << endl;
	cout << CudaOps.all_step_vectors[120][121][2] << endl;
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

