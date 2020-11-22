#pragma once


const int RAYS_PER_DIM = 1600;// 1024; //Max SM's 68
const int RAY_BLOCKS_PER_DIM = 2;
//const float OBJECT_SIZE = 600;
const int NUM_RAYS = RAYS_PER_DIM * RAYS_PER_DIM;
//const float RAY_RANGE = 3.1415 * 0.1;
const float RAY_STEPSIZE = 0.5;
const int RAY_STEPS = 1000;
const int N_STREAMS = 4;
const int THREADS_PER_BLOCK = 256;

const float FOCAL_LEN = 1;	
const int CAM_RADIUS = 512;
const int CAM_RADIUS_INC = 100;


//Dataset specific!!
const int VOL_X = 512;
const int VOL_Y = 512;
const int VOL_Z = 38;
//const int VOL_MAX_WIDTH = sqrt(VOL_X * VOL_X + VOL_Y * VOL_Y + VOL_Z*VOL_Z);



const float CAM_ROTATION_STEP = 2 * 3.1415 / 20; // 20 clicks per rotation


const float CLUSTER_MAX_SEP = 10;
const float HU_MIN = -400;
const float HU_MAX = 300;


