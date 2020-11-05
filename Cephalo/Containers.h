#pragma once

#include <iostream>


using namespace std;

struct Float2 {//float3 taken by cuda
	Float2() {}
	Float2(float x, float y) : x(x), y(y) {}
	float x, y;
};
struct Float3	//float3 taken by cuda
{
	Float3() {};
	Float3(float s) : x(s), y(s), z(s) {}
	Float3(float x, float y, float z) : x(x), y(y), z(z) {}
	float x, y, z;
	void print() { cout << x << " " << y << " " << z << endl; }
	float dot(const Float3 a) { return (float)x * a.x + y * a.y + z * a.z; }
	inline Float3 operator*(float s) const { return Float3(x * s, y * s, z * s); }
	inline Float3 operator-(const Float3& a) const { return Float3(x - a.x, y - a.y, z - a.z); }
	inline Float3 operator+(const Float3& a) const { return Float3(x + a.x, y + a.y, z + a.z); }
};


struct Ray {
	float relative_pitch;
	float relative_yaw;
	Float3 rel_unit_vector;
	Float3 step_vector;	//x, y, z
	Float3 origin;		//x, y, z
	float cam_pitch;
	float cam_yaw;


	float acc_color = 0;	//0..1
	float acc_alpha = 0;	//0..1
	bool full = false;		//

};
struct Block {
	Block() {}
	Block(float c, float a) { color = c; alpha = a; }
	float color = 0.1;
	float alpha = 1;
};

