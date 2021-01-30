#include "CudaContainers.cuh"

Int3::Int3(CudaFloat3 s) { x = s.x; y = s.y; z = s.z; }