#pragma once
#include <string>

const std::string category_names[11] = { "lung", "fat", "fluids", "water", "muscle", 
        "bloodclot", "hematoma", "blood", "cancellous", "cortical", 
        "foreign" };
const int NUM_CATS = 11;
// RAYTRACING
const int RAYS_PER_DIM = 1600;// 1024; //Max SM's 68
const int RAY_BLOCKS_PER_DIM = 2;
//const float OBJECT_SIZE = 600;
const int NUM_RAYS = RAYS_PER_DIM * RAYS_PER_DIM;
//const float RAY_RANGE = 3.1415 * 0.1;
const float RAY_STEPSIZE = 5;
const int RAY_STEPS = 1600;
const int N_STREAMS = 1;
const int THREADS_PER_BLOCK = 512;

// CAMERA SPECIFIC
const float FOCAL_LEN = 1;	
const int CAM_RADIUS = 800;
const int CAM_RADIUS_INC = 100;
const float CAM_ROTATION_STEP = 2 * 3.1415 / 20; // 20 clicks per rotation


//Dataset specific!!
//const int VOL_X = 512;
//const int VOL_Y = 512;
//const int VOL_Z = 163;
//const int VOL_MAX_WIDTH = sqrt(VOL_X * VOL_X + VOL_Y * VOL_Y + VOL_Z*VOL_Z);



// CLUSTERING

const float CLUSTER_MAX_SEP = 10;
const int HU_MIN = -650;
const int HU_MAX = 10000;
const int OUTSIDE_SPECTRUM = -2000;

const int NO_CLUSTER = -1;
const int UNKNOWN_CLUSTER = -2;

const int NON_IGNORES_THRESHOLD = 100;
const float UNKNOWN_CAT = -2;

const float BELONGING_COEFFICIENT = 1 / 0.01;
const float min_belonging = 0.0001;

const int num_K = 12;
const int km_iterations = 1;






// Dont have a good title for this

const float DEFAULT_ALPHA = 0.5;
