# day1
- [day1](#day1)
  - [build cuda code](#build-cuda-code)
  - [cuda programing](#cuda-programing)
  - [pointwiseadd](#pointwiseadd)
  - [ref](#ref)
## build cuda code
- nvcc: NVIDIA CUDA Compiler, 用來編譯 CUDA 程式碼. 
  - nvcc
- performance tools
  - nvprof
  - nvidia-smi
- cuda 有 三種 function
  - `global`: kernel function, can be called from host code
  - `device`: can be called from global or device function
  - `host`: can be called from host code only 
## cuda programing
- cudaMemcpy 有四種方式
  - `cudaMemcpyHostToDevice`: host memory to device memory
  - `cudaMemcpyDeviceToHost`: device memory to host memory
  - `cudaMemcpyDeviceToDevice`: device memory to device memory
  - `cudaMemcpyHostToHost`: host memory to host memory

- cudaMalloc 完成後需要再做 cudaMemcpy 才能把資料從 host memory 複製到 device memory.
- 先 malloc memory 再做 cudaMemcpy, 這樣可以避免在 kernel function 中使用 host memory.
- `cudaDeviceSynchronize()` 用來確保所有的 kernel function 都已經執行完畢, 這樣可以避免在 kernel function 中使用未完成的資料.
- `cudaGetLastError()` 用來檢查 CUDA 驅動是否返回了任何異常, 這樣可以避免在 kernel function 中使用未完成的資料.
- cudaMemcpy 需要指定資料的方向, 例如 cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost 等等.
- cudaMalloc 需要指定資料的大小, 例如 cudaMalloc(&A_gpu, n * sizeof(int)).
- cudaMemcpy 需要指定資料的大小, 例如 `cudaMemcpy(A_gpu, A, n * sizeof(int), cudaMemcpyHostToDevice).`
## pointwiseadd 
```c
#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>

__global__ void pointwise_add_kernel(int* C, const int* A, const int* B, int n) {
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

int main() {
    const int n = 128;
    int* C = new int[n];
    int* A = new int[n];
    int* B = new int[n];
    for (int i = 0; i < n; ++i) {
        A[i] = i;
        B[i] = i*i;
    }
    // need to use cudamalloc
    int* A_gpu, *B_gpu, *C_gpu;
    cudaMalloc(&A_gpu, n * sizeof(int));
    cudaMalloc(&B_gpu, n * sizeof(int));
    cudaMalloc(&C_gpu, n * sizeof(int));
    cudaMemcpy(A_gpu, A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * sizeof(int), cudaMemcpyHostToDevice);
    pointwise_add_kernel<<<1, 1>>>(C_gpu, A_gpu, B_gpu, n);
    /***********************/
    //pointwise_add_kernel<<<1, 1>>>(C, A, B, n);
    cudaDeviceSynchronize();    // 见下方 Aside
    cudaError_t error = cudaGetLastError(); // 检查当前 CUDA 驱动是否返回了任何异常。调用这句话之前记得调用 cudaDeviceSynchronize()
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    // Copy the result from C_gpu to C
    cudaMemcpy(C, C_gpu, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        assert(C[i] == A[i] + B[i]);
    	assert(C[i] == A[i] - B[i]);
    
    }
    return 0;
}

```

```bash 
nvcc -c -O3 pointwise.cu -o pointwise.o
```
## ref
- https://github.com/interestingLSY/CUDA-From-Correctness-To-Performance-Code
- https://hpcwiki.io/gpu/cuda/#:~:text=,%E6%B4%BB%E8%B7%83%E6%97%B6%E9%97%B4%E6%AF%94%E3%80%81%E6%B4%BB%E8%B7%83%E7%9A%84%20SM%20%E6%AF%94%E4%BE%8B%E7%AD%89%E7%AD%89%EF%BC%89%E6%9D%A5%E5%B8%AE%E5%8A%A9%E4%BD%A0%E6%9B%B4%E5%8A%A0%E7%BB%86%E8%87%B4%E5%9C%B0%E4%BC%98%E5%8C%96%20CUDA%20Kernel%E3%80%82