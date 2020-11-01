#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;

const int RAYS_PER_DIM = 512;
const float OBJECT_SIZE = 512;
const int NUM_RAYS = RAYS_PER_DIM * RAYS_PER_DIM;
//const float RAY_RANGE = 3.1415 * 0.1;
const float RAY_STEPSIZE = 1;
const float CAMERA_RADIUS = 1;
const int FOCAL_LEN = 50;	// Only implemented in rendering so far!!

const int VOL_X = 256;
const int VOL_Y = 256;
const int VOL_Z = 30;

const int IM_SIZE = 256;
//float camera_dist = 512;

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
	float dot(const Float3 a) {return (float) x * a.x + y * a.y + z * a.z; }
	inline Float3 operator*(float s) const { return Float3(x * s, y * s, z * s); }
	inline Float3 operator-(const Float3& a) const { return Float3(x - a.x, y - a.y, z - a.z); }
	inline Float3 operator+(const Float3& a) const { return Float3(x + a.x, y + a.y, z + a.z); }
};


struct Ray {
	float relative_pitch;
	float relative_yaw;
	Float3 step_vector;	//x, y, z
	Float3 origin;		//x, y, z
	
	int render_x;
	int render_y;

	float acc_color = 0;	//0..1
	float acc_alpha = 0;	//0..1
	bool full = false;		//

	void reset(float x, float y, float z, float x_, float y_, float z_) { // origin, step_vector
		acc_color = 0;
		acc_alpha = 0;
		full = false;
		origin = Float3(x, y, z);
		step_vector = Float3(x_, y_, z_);
	}
};
struct Block {
	Block() {}
	Block(float c, float a) { color = c; alpha = a; }
	float color;
	float alpha=1;
};

class Volume {
public:
	Block* blocks;
	//Volume() {blocks = (Block*) malloc (512 * 512 * 30 * sizeof(Block)); }
	Volume() {
		cout << "Doing this " << endl;
		blocks = new Block[512 * 512 * 30];
		for (int i = 0; i < 512 * 512 * 30; i++) {
			blocks[i] = Block();
		}
	}
		
	int xyzToIndex(int x, int y, int z) { return z * 512 * 512 + y * 512 + x; }
	void testSetup() {
		cout << sizeof(Block) << endl;
		for (int z = 0; z < 30; z++) {
			for (int y = 0; y < 512; y++) {
				for (int x = 0; x < 512; x++) {
					cout << x <<" " << y << " " << z << " "<< xyzToIndex(x, y, z) << endl;
					blocks[xyzToIndex(x, y, z)].color = x / 512;
					//blocks[xyzToIndex(x, y, z)] = Block(x / 512, 1);
					//blocks[1] = Block(23, 1);
					//blocks[xyzToIndex(x, y, z)].alpha = 1 * (x < 128);
				}
			}
		}
	}
};

