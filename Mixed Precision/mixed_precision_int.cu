#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sm_61_intrinsics.h>
#include <helper_timer.h>
#include <helper_math.h>
#include <cooperative_groups.h>
#include <cstdio>

#include "util.cuh"

using namespace cooperative_groups;

// FMA numerical arithmetic function in GPU @INT8
// y = x * y + z
// Assuming we have transposed matrix y
__global__ void dp4a_kernel(char* d_x, char* d_y, int* d_z, int size)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	char4* quad_x = (char4*)d_x;
	char4* quad_y = (char4*)d_y;

	for (int i = idx_x; i < size; i += stride)
	{
		d_z[i] = __dp4a(quad_y[i], quad_x[i], 0);
	}
}

void dp4a_host(char* h_x, char* h_y, int* h_z, int size)
{
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		int sum = 0;
		for (int j = 0; j < 4; j++)
		{
			sum += (int)h_y[4 * i + j] * (int)h_x[4 * i + j];
		}
		h_z[i] = sum;
	}
}

/*int main()
{
	CBuffer<char> X, Y;
	CBuffer<int> Z;
	int size = 1 << 26;

	srand(2021);

	// Initialize host buffers
	X.init(size, true);
	Y.init(size, true);
	Z.init(size, true);

	// Initialize GPU buffers
	X.cuda();
	Y.cuda();
	Z.cuda();

	// Get number of blocks for grid-stride loop
	int n_threads = 256;
	int num_sms;
	int num_blocks_per_sm;

	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, dp4a_kernel, n_threads, n_threads * sizeof(int));

	int n_blocks = std::min(num_blocks_per_sm * num_sms, (size / 4 + n_threads - 1) / n_threads);

	// Initialize timer
	StopWatchInterface* timer;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	dp4a_kernel << <n_blocks, n_threads, n_threads * sizeof(float2) >> > (X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size);

	cudaDeviceSynchronize();
	sdkStopTimer(&timer);

	float elapsedTimeMS = sdkGetTimerValue(&timer);
	float ops = size / elapsedTimeMS * 1e-6;
	printf("IMA, FLOPS = %.3f GFlops, Operation Time= %.3f msec\n", ops, elapsedTimeMS);

	dp4a_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size);

	int diff_count = Z.diff_count();
	(diff_count == 0) ? printf("Success\n") : printf("Counted %d differences\n", diff_count);

	// Cleanup
	sdkDeleteTimer(&timer);

	return 0;
}*/
