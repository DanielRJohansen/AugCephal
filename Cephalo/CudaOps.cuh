#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class CudaOperator {
	CudaOperator() {};

private:

	void showDevices();
};