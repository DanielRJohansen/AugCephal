#pragma once
#include "constants.h"
#include <iostream>
#include <vector>
#include <string>

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
	float dot(const Float3 a) { return (float)x * a.x + y * a.y + z * a.z; }
	inline Float3 operator*(float s) const { return Float3(x * s, y * s, z * s); }
	inline Float3 operator-(const Float3& a) const { return Float3(x - a.x, y - a.y, z - a.z); }
	inline Float3 operator+(const Float3& a) const { return Float3(x + a.x, y + a.y, z + a.z); }
};

struct Color {
	Color() {};
	Color(float red, float gre, float blu) { r = red; g = gre; b = blu; };
	float r;
	float g;
	float b;
	inline Color operator*(float s) const { return Color(r * s, g * s, b * s); }
	inline Color operator+(Color a) const { return Color(r + a.r, g + a.g, b + a.b); }
	Color add(Color c);
	Color mul(float s);
	Color cutOff(Color c);
};

struct Ray {
	Float3 rel_unit_vector;
};

struct Cluster {
	Cluster(int i, int s) { id = i; size = s; };
	int id;
	int size;
};



struct Category {
	Category() {};
	Category(int id, int cc, int start, int stop, Color c) : id(id), cat_cluster(cc), start(start), stop(stop), color(c) {
		center = (start + stop) / 2;
		spread = stop - start;
	};

	int id;
	int cat_cluster;
	int start;
	int stop;
	int center;
	int spread;
	Color color;
};

struct Block {
	int hu_val;
	int cat;
	int cat_;

	Color color;
	float alpha = 1;	//Basically how many cat switches we allow for
	bool ignore = false;
	bool was_ignore = false;
	
	int cluster_id = NO_CLUSTER;// No cluster
};


struct ColorScheme {
	ColorScheme();
	Category *categories;
	int upper_limit = HU_MAX;
	int lower_limit = HU_MIN;
	Color colors[400 + 300];
	int cat_indexes[400 + 300];
	string category_ids[6] = { "lung", "fat", "fluids", "muscle", "clot", "bone" };


};

struct CompactCam {
	CompactCam(Float3 origin, float pitch, float yaw, float r);
	Float3 origin;
	float radius;
	float sin_pitch;
	float cos_pitch;
	float sin_yaw;
	float cos_yaw;
};