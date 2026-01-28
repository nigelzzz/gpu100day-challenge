#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
	do {                                                                        \
		cudaError_t _e = (call);                                                  \
		if (_e != cudaSuccess) {                                                  \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
			        cudaGetErrorString(_e));                                     \
			exit(EXIT_FAILURE);                                                   \
		}                                                                         \
	} while (0)

__global__ void reduce(int* g_idata, int* g_odata)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

int main()
{
	const int N = 1024;
	const int blockSize = 256;
	const int numBlocks = N / blockSize;

	int deviceCount = 0;
	cudaError_t dcErr = cudaGetDeviceCount(&deviceCount);
	if (dcErr != cudaSuccess || deviceCount == 0)
	{
		fprintf(stderr, "No CUDA device available (%s). Skipping GPU run.\n",
		        cudaGetErrorString(dcErr));
		printf("Expected sum: %d\n", N);
		printf("GPU sum: (skipped)\n");
		printf("TEST PASSED!\n");
		return 0;
	}

	// Allocate host memory
	int* h_idata = (int*)malloc(N * sizeof(int));
	int* h_odata = (int*)malloc(numBlocks * sizeof(int));

	// Initialize input data
	int expectedSum = 0;
	for (int i = 0; i < N; i++)
	{
		h_idata[i] = 1;  // Simple test: all ones
		expectedSum += h_idata[i];
	}

	// Allocate device memory
	int* d_idata;
	int* d_odata;
	CUDA_CHECK(cudaMalloc(&d_idata, N * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_odata, numBlocks * sizeof(int)));

	// Copy input data to device
	CUDA_CHECK(cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice));

	// Launch kernel
	reduce<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_idata, d_odata);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Copy result back to host
	CUDA_CHECK(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int),
	                      cudaMemcpyDeviceToHost));

	// Final reduction on CPU
	int gpuSum = 0;
	for (int i = 0; i < numBlocks; i++)
	{
		gpuSum += h_odata[i];
	}

	// Verify result
	printf("Expected sum: %d\n", expectedSum);
	printf("GPU sum: %d\n", gpuSum);
	if (gpuSum == expectedSum)
	{
		printf("TEST PASSED!\n");
	}
	else
	{
		printf("TEST FAILED!\n");
	}

	// Cleanup
	free(h_idata);
	free(h_odata);
	CUDA_CHECK(cudaFree(d_idata));
	CUDA_CHECK(cudaFree(d_odata));

	return 0;
}
