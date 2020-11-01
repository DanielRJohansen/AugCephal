#include "Raytracing.h"


void Raytracer::initRaytracer(Camera *c) { 
	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	camera = c;
	initRays();
	blocks = new Block[512 * 512 * 30];
	CudaOps.newVolume(blocks);
	cout << "Volume size " << 512*512*30*sizeof(Block)/1000000. << " MB" << endl;
	cout << "Raytracer Initialized" << endl;
}
Raytracer::~Raytracer() {}


void Raytracer::initRays() {
	float ray_range = asin((OBJECT_SIZE / 2. )/ camera->radius) * 2;
	float ray_increment = ray_range / RAYS_PER_DIM;
	float start_angle = - ray_range / 2. + ray_increment / 2.;	// Shift by half increment to have
													// same amount of rays above and below 0
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		float relative_pitch = start_angle + ray_increment * y;
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float relative_yaw = - start_angle - ray_increment * x;	// Flip the angles here!
			Ray ray;
			ray.relative_pitch = relative_pitch;
			ray.relative_yaw = relative_yaw;
			rayptr[xyToRayIndex(x, y)] = ray;
		}
	}
}





cv::Mat Raytracer::render() {
	cout << "Rendering" << endl;
	precalcSinCos();
	castRays();
	catchRays();
	return projectRaysOnPlane();	//TODO: only calculate rays belonging to pixels,
									// if focal point has changed.
}


void Raytracer::precalcSinCos() {
	for (int i = 0; i < RAYS_PER_DIM; i++) {	// The indexing here is lazy, but it works \_O
		sin_pitches[i] = sin(rayptr[xyToRayIndex(i, i)].relative_pitch + camera->plane_pitch);
		cos_pitches[i] = cos(rayptr[xyToRayIndex(i, i)].relative_pitch + camera->plane_pitch);
		sin_yaws[i] = sin(rayptr[xyToRayIndex(i, i)].relative_yaw + camera->plane_yaw);
		cos_yaws[i] = cos(rayptr[xyToRayIndex(i, i)].relative_yaw + camera->plane_yaw);
	}
}

void Raytracer::castRays() {
	for (int y = 0; y < RAYS_PER_DIM; y++) {
		for (int x = 0; x < RAYS_PER_DIM; x++) {
			float x_ = 1 * sin_pitches[y] * cos_yaws[x];
			float y_ = 1 * sin_pitches[y] * sin_yaws[x];
			float z_ = 1 * cos_pitches[y];
			
			rayptr[xyToRayIndex(x, y)].step_vector = Float3(x_ * RAY_STEPSIZE,
				y_ * RAY_STEPSIZE, z_ * RAY_STEPSIZE);
			rayptr[xyToRayIndex(x, y)].origin = camera->origin;
		}
	}
	CudaOps.rayStep(rayptr);
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
	cv::Mat image = cv::Mat::zeros(cv::Size(RAYS_PER_DIM, RAYS_PER_DIM), CV_8U);
	for (int y = 0; y < 512; y++) {
		for (int x = 0; x < 512; x++) {
			image.at<uchar>(y, x) = rayptr[xyToRayIndex(x, y)].acc_color * 256;
		}
	}
	return image;
}

