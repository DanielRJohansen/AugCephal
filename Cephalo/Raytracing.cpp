#include "Raytracing.h"

Ray::Ray(float rel_pitch, float rel_yaw) {
	relative_pitch = rel_pitch;
	relative_yaw = rel_yaw;	
}


//void Ray::castRay(float3 cam_pos, RayInfo RF ) {
//	origin = cam_pos;
//
//	makeStepVector(RF);
//}

void Ray::makeStepVector(RayInfo RF) {
	float x = radius * RF.sin_pitch * RF.cos_yaw;
	float y = radius * RF.sin_pitch * RF.sin_yaw;
	float z = radius * RF.cos_pitch;
	
	// Step vector
	//step_vector = float3(x, y, z);
	//step_vector = step_vector * RAY_STEPSIZE;
}

void Ray::rayStep(int step) {
	//float3 pos = origin + step_vector * step;
	//cout << pos.x << endl;
}






Raytracer::Raytracer() { 
	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	initRays();
	cout << "Raytracer Initialized" << endl;

}
Raytracer::~Raytracer() {}

cv::Mat Raytracer::render(Camera camera) {
	//castRays(camera);
	catchRays();
	return projectRaysOnPlane();	//TODO: only calculate rays belonging to pixels,
									// if focal point has changed.
}

void Raytracer::initRays() {
	float ray_increment = RAY_RANGE / RAYS_PER_DIM;
	float start_angle = - RAY_RANGE / 2. + ray_increment / 2.;	// Shift by half increment to have
													// same amount of rays above and below 0
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		float relative_pitch = start_angle + ray_increment * y;
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float relative_yaw = start_angle + ray_increment * x;
			rayptr[xyToIndex(x, y)] = Ray(relative_pitch, relative_yaw);
		}
	}
}

/*
void Raytracer::castRays(Camera camera) {
	for (int i = 0; i < RAYS_PER_DIM; i++) {
		sin_pitches[i] = sin(rayptr[xyToIndex(0,i)].relative_pitch + camera.plane_pitch);
		cos_pitches[i] = cos(rayptr[xyToIndex(0, i)].relative_pitch + camera.plane_pitch);
		sin_yaws[i] = sin(rayptr[xyToIndex(i, 0)].relative_yaw + camera.plane_yaw);
		cos_yaws[i] = cos(rayptr[xyToIndex(i, 0)].relative_yaw + camera.plane_yaw);
	}
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			//rayptr[xyToIndex(x, y)].castRay(float3(camera.x, camera.y, camera.z),
			//	RayInfo(sin_pitches[y], cos_pitches[y], sin_yaws[x], cos_yaws[x]));
		}
	}
	cout << "Rays cast" << endl;
}*/

void Raytracer::catchRays() {
	for (int i = 200; i < 500; i++) {
		cout << i << endl;
		for (int j = 0; j < NUM_RAYS; j++) {
			rayptr[j].rayStep(i);
		}
	}
}

cv::Mat Raytracer::projectRaysOnPlane() {
	cv::Mat image;
	return image;
}

