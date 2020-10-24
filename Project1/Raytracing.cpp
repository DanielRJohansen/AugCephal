#include "Raytracing.h"

Ray::Ray(float rel_pitch, float rel_yaw, float ss) {
	relative_pitch = rel_pitch;
	relative_yaw = rel_yaw;
	stepsize = ss;
	
}

void Ray::castRay(float x, float y, float z, RayInfo RF ) {
	x_origin = x;
	y_origin = y;
	z_origin = z;

	makeStepVector(RF);
}

void Ray::makeStepVector(RayInfo RF) {
	
	float x = radius * RF.sin_pitch * RF.cos_yaw;
	float y = radius * RF.sin_pitch * RF.sin_yaw;
	float z = radius * RF.cos_pitch;
	
	// Unit vector
	x = x - x_origin;
	y = y - y_origin;
	z = z - z_origin;

	float dist = sqrt(x * x + y * y + z * z);
	cout << dist << endl;

	//cout << x << "  " << y << "  " << z;
}






Raytracer::Raytracer() { 
	cout << "we here " << endl; 
	//initRays();
	//a = 2;
}
Raytracer::~Raytracer() {}
cv::Mat Raytracer::render(Camera camera) {
	castRays(camera);
	catchRays();
	return projectRaysOnPlane();	//TODO: only calculate rays belonging to pixels,
									// if focal point has changed.
}

void Raytracer::initRays() {
	cout << "Making rays" << endl;
	float ray_increment = RAY_RANGE / RAYS_PER_DIM;
	float start_angle = RAY_RANGE / 2 + ray_increment / 2;	// Shift by half increment to have
													// same amount of rays above and below 0
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		float relative_pitch = start_angle + ray_increment * y;
		
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float relative_yaw = start_angle + ray_increment * x;
			rays[y][x] = Ray(relative_pitch, relative_yaw, RAY_STEPSIZE);
		}
	}
}

void Raytracer::castRays(Camera camera) {
	for (int i = 0; i < RAYS_PER_DIM; i++) {
		sin_pitches[i] = sin(rays[i][0].relative_pitch + camera.plane_pitch);
		cos_pitches[i] = cos(rays[i][0].relative_pitch + camera.plane_pitch);
		sin_yaws[i] = sin(rays[0][i].relative_yaw + camera.plane_yaw);
		cos_yaws[i] = cos(rays[0][i].relative_yaw + camera.plane_yaw);
	}
}

void Raytracer::catchRays() {
	int i = 0;
}

cv::Mat Raytracer::projectRaysOnPlane() {
	cv::Mat image;
	return image;
}

