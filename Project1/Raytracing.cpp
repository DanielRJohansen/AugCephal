#include "Raytracing.h"

Ray::Ray(Camera camera, float relative_pitch, float relative_yaw, float stepsize) {
	x_origin = camera.x;
	y_origin = camera.y;
	z_origin = camera.z;
	stepsize = stepsize;
	makeStepVector(camera.pitch + relative_pitch, camera.yaw + relative_yaw);
}

void Ray::makeStepVector(float tilt, float yaw) {
	float y_ = 1;
	float x_, z_ = 0;
	// Rotate around z axis --optimized
	x_ = sin(-yaw)*y_;
	y_ = cos(-yaw)*y_;
	cout << yaw << endl;
	// Rotate around x axis --optimized
	y_ = cos(-tilt)*y_;
	z_ = -sin(-tilt)*y_;

	/*x_ = x_origin + x_;
	y_ = y_origin + y_;
	z_ = z_origin + z_;
	*/
	// Step-vector 
	//x_
	cout << x_ << "  " << y_ << "  " << z_;
}
