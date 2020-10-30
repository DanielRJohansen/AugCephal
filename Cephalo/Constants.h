#pragma once
const int RAYS_PER_DIM = 256;
const int NUM_RAYS = RAYS_PER_DIM* RAYS_PER_DIM;
const float RAY_RANGE = 3.1415 * 0.5;
const float RAY_STEPSIZE = 1;
const float CAMERA_RADIUS = 1;
//float camera_dist = 512;

/*
struct float3
{
	float3() {};
	float3(float s) : x(s), y(s), z(s) {}
	float3(float x, float y, float z) : x(x), y(y), z(z) {}
	float x, y, z;

	inline float3 operator*(float s) const { return float3(x * s, y * s, z * s); }
	inline float3 operator+(const float3& a) const { return float3(x + a.x, y + a.y, z + a.z); }
};*/


struct Ray {
	float relative_pitch;
	float relative_yaw;
	float step_vector[3];	//x, y, z
	float origin[3];		//x, y, z

	void reset(float x, float y, float z, float x_, float y_, float z_) {
		acc_color = 0;
		acc_alpha = 0;
		full = false;
		origin[0] = x;
		origin[1] = y;
		origin[2] = z;
		step_vector[0] = x_;
		step_vector[1] = y_;
		step_vector[2] = z_;
	}

	float acc_color = 0;	//0..1
	float acc_alpha = 0;	//0..1
	bool full = false;		//
};
struct Block {
	float color;
	float alpha;
};
struct Volume {
	Block blocks[256][256][30];
	void testSetup() {
		for (int z = 0; z < 30; z++) {
			for (int y = 0; y < 256; y++) {
				for (int x = 0; x < 256; x++) {
					blocks[x][y][z].alpha = 1;
					blocks[x][y][z].alpha = 1 * (x < 128);
				}
			}
		}
	}
};