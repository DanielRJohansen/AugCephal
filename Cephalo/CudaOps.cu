#include "CudaOps.cuh"

#define vol_x_range VOL_X
#define vol_y_range VOL_Y
#define vol_z_range VOL_Z
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void squareKernel(int* a)
{
    int i = blockIdx.x * 512 + threadIdx.x;
    a[i] = i;
}

__global__ void stepKernel(Ray* rayptr, Block *blocks, bool *success, float* coor) {
    int index = blockIdx.x * RAYS_PER_DIM + threadIdx.x;  //This fucks shit up if RPD > 1024!!
    

    for (int step = 0; step < 2000; step++) {
        if (true){//!rayptr[index].full) {
            float x_ = rayptr[index].origin.x + rayptr[index].step_vector.x * step;
            float y_ = rayptr[index].origin.y + rayptr[index].step_vector.y * step;
            float z_ = rayptr[index].origin.z + rayptr[index].step_vector.z * step;
            int vol_x = (int)x_ + vol_x_range / 2;
            int vol_y = (int)y_ + vol_y_range / 2;
            int vol_z = (int)z_ + vol_z_range / 2;
            if (index == 0) {
                coor[step] = z_;
            }
                
            if (vol_x >= 0 && vol_y >= 0 && vol_z >= 0 && // Only proceed if coordinate is within volume!
                vol_x < vol_x_range && vol_y < vol_y_range && vol_z < vol_z_range) {
                int volume_index = vol_z * 512 * 512 + vol_y * 512 + vol_x;
                rayptr[index].acc_color += blocks[volume_index].color;
                rayptr[index].acc_alpha += blocks[volume_index].alpha;
                if (rayptr[index].acc_alpha >= 1) 
                    rayptr[index].full = true;
            }
        }
    }
    * success = 1;
}


CudaOperator::CudaOperator(){
    //cout << (NUM_RAYS * sizeof(Ray))/1000000. << " MB" << "  " << sizeof(Block)/1000000. << " MB" <<endl;
    cudaMallocManaged(&rayptr, NUM_RAYS * sizeof(Ray));
    cudaMallocManaged(&t, 50 * sizeof(testObject));
    cudaMallocManaged(&blocks, 512*512*30*sizeof(Block));
    cout << "Cuda initialized" << endl;
    }
void CudaOperator::newVolume(Block* bs) { 
    cudaMemcpy(blocks, bs, 512*512*30 * sizeof(Block), cudaMemcpyHostToDevice);
    //Volume only needs to go one way as it is not altered.
}


void CudaOperator::rayStep(Ray *rp) {
    cudaMemcpy(rayptr, rp, NUM_RAYS * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    bool* success;
    cudaMallocManaged(&success, sizeof(bool));
    *success = false;
    float* coor;
    cudaMallocManaged(&coor, 2000*sizeof(float));

    stepKernel << <RAYS_PER_DIM, RAYS_PER_DIM >> > (rayptr, blocks, success, coor);    // RPD blocks (y), RPD threads(x)
    cudaDeviceSynchronize();
    cout << "Success: " << *success << endl;
    cout << rayptr[0].origin.y << endl;
    for (int i = 0; i < 100; i++) {
        //cout << rayptr[i].acc_color << endl;
        cout << coor[i] << " ";
    }
    cout << endl;
    //cout << rayptr[120].acc_color << "  " << rayptr[140].acc_color;
    cout << "Raystep complete "<< endl;

    //Finally the CUDA altered rayptr must be copied back to the Raytracer rayptr
    cudaMemcpy(rp, rayptr, NUM_RAYS * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
}
















void CudaOperator::doStuff() {
    int* a;
    cudaMallocManaged(&a, 50*512 * sizeof(int));

    squareKernel << <50, 512 >> > (a);
    cudaDeviceSynchronize();
    for (int i = 0; i < 50*512; i++) {
        cout << a[i] << " ";
    }
    cudaFree(a);
}

__global__ void testKernel(testObject *t, float* a, bool *finished)
{
    int i = threadIdx.x;
    a[i] = t[i].var*2;
    t[i].var = t[i].var / 2;
    *finished = true;
}
void CudaOperator::objectTesting(testObject* te) {
    //t = te;
    cudaMemcpy(t, te, 50 * sizeof(testObject), cudaMemcpyHostToDevice);
    cout << t[0].var << endl;
    float* a;
    cudaMallocManaged(&a, 50 * sizeof(float));

    bool* finished;
    cudaMallocManaged(&finished, sizeof(bool));
    *finished = false;
    testKernel << <1, 50 >> > (t, a, finished);
    cudaDeviceSynchronize();
    cout <<"Finished: " << *finished << endl;
    for (int i = 0; i < 50; i++) {
        cout << t[i].var << " ";
    }
    cout << endl;
    cudaMemcpy(te, t, 50 * sizeof(testObject), cudaMemcpyDeviceToHost);


}