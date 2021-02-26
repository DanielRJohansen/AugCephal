#include "Resizer.h"


using namespace std;

double cubicInterpolate(double p[4], double x) {
	return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

double bicubicInterpolate(double p[4][4], double x, double y) {
	double arr[4];
	arr[0] = cubicInterpolate(p[0], y);
	arr[1] = cubicInterpolate(p[1], y);
	arr[2] = cubicInterpolate(p[2], y);
	arr[3] = cubicInterpolate(p[3], y);
	return cubicInterpolate(arr, x);
}

double tricubicInterpolate(double p[4][4][4], double x, double y, double z) {
	double arr[4];
	arr[0] = bicubicInterpolate(p[0], y, z);
	arr[1] = bicubicInterpolate(p[1], y, z);
	arr[2] = bicubicInterpolate(p[2], y, z);
	arr[3] = bicubicInterpolate(p[3], y, z);
	return cubicInterpolate(arr, x);
}

float* Resizer::getKernel(int x, int y) {
	float* kernel = new float[16];

	int i = 0;
	for (int y_ = y; y_ < y + 4; y_++) {
		for (int x_ = x; x_ < x + 4; x_++) {
			if (isLegal(x_, y_, size_from)) {
				kernel[i] = from_slice[xyToIndex(x_, y_, size_from)];
			}
			else
				kernel[i] = 0;
			i++;
		}
	}
	return kernel;
} 


float* Resizer::Interpolate2D(float* slice) {	// Can only handle doublling the size
	to_slice = new float[size_to * size_to];
	//assert(size_from * 2 == size_to);
	from_slice = slice;
	printf("From: %d    To: %d\n", size_from, size_to);
	double kernel[4][4];
	for (int y = -2; y < size_from; y++) {		// On purpose we only start 2 pixels out of frame, NOT end
		for (int x = -2; x < size_from; x++) {
			
				
			//if (isLegal(x, y, size_from)) {
			//	to_slice[xyToIndex(x*2, y*2, size_to)] = slice[xyToIndex(x, y, size_from)];
			//}
			float* kernel_ = getKernel(x, y);
			
			for (int i = 0; i < 16; i++) {
				kernel[i / 4][i % 4] = (double) kernel_[i];
			}
			

			for (int yoff = 0; yoff < 2; yoff++) {
				for (int xoff = 0; xoff < 2; xoff++) {
					if (isLegal(x*2 + xoff, y*2 + yoff, size_to)) {
						double point_val = bicubicInterpolate(kernel, 2/6. + xoff/6., 2/6. +yoff/6.);
						//double point_val = bicubicInterpolate(kernel, 2 / 7. + xoff / 7., 2 / 7. + yoff / 7.);
						//double point_val = bicubicInterpolate(kernel, 0, 0);
						int point_index = xyToIndex(x*2 + xoff, y*2 + yoff, size_to);
						to_slice[point_index] = point_val;
						//to_slice[point_index] = slice[xyToIndex(x, y, size_from)];
					}
					
				}
			}			
			delete(kernel_);
		}
	}
	return to_slice;
}
void Resizer::getKernel(float* kernel, int x, int y, int z) {
	int i = 0;
	for (int z_ = z; z_ < z + 4; z_++) {
		for (int y_ = y; y_ < y + 4; y_++) {
			for (int x_ = x; x_ < x + 4; x_++) {
				if (isLegal(x_, y_, z_, width_old, height_old)) {
					float a = raw_scans[xyzToIndex(x_, y_, z_, width_old, height_old)];
					kernel[i] = a;
				}
				else
					kernel[i] = 0;
				i++;
			}
		}
	}
}

Int3 Resizer::Interpolate3D(float* rs, float* resized_scan, int xy_from, int xy_to, int num_slices, float z_over_xy) {
	int num_slices_ = (int)num_slices * z_over_xy;
	width_old = xy_from; width_new = xy_to; height_old = num_slices; height_new = num_slices_;

	raw_scans = rs;
	resized_scan = new float[num_slices_ * xy_to * xy_to];
	printf("Resizing from:		%d %d %d   to %d %d %d\n", height_old, width_old, width_old, height_new, width_new, width_new);

	float* kernel = new float[4 * 4 * 4];
	double kernel_arr[4][4][4];
	int z_new = 0;
	float z_new_mapped = z_new / z_over_xy;

	for (int z = -1; z <= height_old; z++) {
		printf("Z: %d\r", z);
		for (int y = 0; y < width_old; y++) {	// Honestly we dont really care about the edges here, so lets just save some computation.
			for (int x = 0; x < width_old; x++) {
				getKernel(kernel, x, y, z);
				for (int i = 0; i < 64; i++) {
					kernel_arr[i/16][(i % 16) / 4][i % 4] = kernel[i];
				}
				
				while (z_new / z_over_xy < (float)(z + 2)) {
					z_new_mapped = z_new / z_over_xy;
					float rel_z = (z_new_mapped - z) / 3.;	//Image on phone for explanation
					

					for (int yoff = 0; yoff < 2; yoff++) {
						for (int xoff = 0; xoff < 2; xoff++) {
							if (isLegal(x * 2 + xoff, y * 2 + yoff, size_to)) {
								double point_val = tricubicInterpolate(kernel_arr, 2 / 6. + xoff / 6., 2 / 6. + yoff / 6., rel_z);
								int point_index = xyzToIndex(x * 2 + xoff, y * 2 + yoff, z_new, width_new, height_new);
								resized_scan[point_index] = point_val;
							}
						}
					}
					z_new++;
				}
			}
		}
	}
	delete(kernel);
	printf("\nResizing completed\n");
	return Int3(width_new, width_new, height_new);
}


