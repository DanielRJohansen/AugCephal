#include "Raytracing.h"






void Raytracer::initRaytracer(Camera c) { 
	rayptr = (Ray*)malloc(NUM_RAYS * sizeof(Ray));
	camera = c;
	initRays();
	initCuda();	
	volume.testSetup();

	precalcSinCos();
	castRays();
	catchRays();
	initRenderPlane();
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
		//cout << camera.focal_plane_point.x << " " << rayptr[i].origin.x << " " << rayptr[i].step_vector.x << end;
		float ray_dist = (camera.focal_plane_point + rayptr[i].origin).dot((rayptr[i].step_vector*(1/RAY_STEPSIZE)));
		cout << "Ray dist  " << ray_dist << endl;
		cout << rayptr[i].step_vector.x << " " << rayptr[i].step_vector.y << " " << rayptr[i].step_vector.z << endl;
		Float3 ray_pos_on_plane = rayptr[i].origin + rayptr[i].step_vector * ray_dist;
		Float2 rp_pos = convertGlobalCoorToRenderCoor(ray_pos_on_plane);
		cout << rp_pos.x << " " << (int)rp_pos.x << endl;
		rayptr[i].render_x = (int)rp_pos.x;
		rayptr[i].render_y = (int)rp_pos.y;
		cout << rayptr[i].render_y << " " << rayptr[i].render_x << endl;
		rendering.raycnt[rayptr[i].render_y][rayptr[i].render_y] += 1;
		for (int y = 0; y < 256; y++) {
			for (int x = 0; x < 256; x++) {
				cout << rendering.raycnt[y][x] << " ";
			}
			cout << endl;
		}	
	}
}

Float2 Raytracer::convertGlobalCoorToRenderCoor(Float3 glob) {
	return Float2(glob.x, glob.z);	// Quick implementation, only works for initial camera pos
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
			
			rayptr[xyToRayIndex(x, y)].step_vector = Float3(x_ * RAY_STEPSIZE,
				y_ * RAY_STEPSIZE, z_ * RAY_STEPSIZE);
			//rayptr[xyToRayIndex(x, y)].step_vector[0] = x_ * RAY_STEPSIZE;
			//rayptr[xyToRayIndex(x, y)].step_vector[1] = y_ * RAY_STEPSIZE;
			//rayptr[xyToRayIndex(x, y)].step_vector[2] = z_ * RAY_STEPSIZE;

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

