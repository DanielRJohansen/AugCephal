#pragma once


const int RAYS_PER_DIM = 1024;
//const float OBJECT_SIZE = 600;
const int NUM_RAYS = RAYS_PER_DIM * RAYS_PER_DIM;
//const float RAY_RANGE = 3.1415 * 0.1;
const float RAY_STEPSIZE = 0.5;
const int RAY_STEPS = 1000;
const float CAMERA_RADIUS = 1;
const float FOCAL_LEN = 0.8;	


//Dataset specific!!
const int VOL_X = 512;
const int VOL_Y = 512;
const int VOL_Z = 61;

const int CAM_RADIUS = 512;


const float CAM_ROTATION_STEP = 2 * 3.1415 / 20; // 20 clicks per rotation


const float CLUSTER_MAX_SEP = 10;
const float HU_MIN = -400;
const float HU_MAX = 300;


