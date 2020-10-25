#pragma once
const int RAYS_PER_DIM = 256;
const int NUM_RAYS = RAYS_PER_DIM* RAYS_PER_DIM;
const float RAY_RANGE = 3.1415 * 0.5;
const float RAY_STEPSIZE = 1;

//float camera_dist = 512;



struct float3
{
	float3() {};
	float3(float s) : x(s), y(s), z(s) {}
	float3(float x, float y, float z) : x(x), y(y), z(z) {}
	float x, y, z;

	inline float3 operator*(float s) const { return float3(x * s, y * s, z * s); }
	inline float3 operator+(const float3& a) const { return float3(x + a.x, y + a.y, z + a.z); }
};