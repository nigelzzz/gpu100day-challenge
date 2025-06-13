#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                 \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n",             \
                    cudaGetErrorString(err), __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

const size_t N      = 1 << 24;   // 16,777,216
const int    REPEAT = 100;       // 每個 blockSize 重複跑 100 次取平均
const int    INNER  = 10000; 
// ---------------- Kernel ----------------
__global__ void vecAdd(const float* a, const float* b, float* c, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
	//c[idx] = a[idx] * b[idx] + c[idx];
}

__global__ void heavyFma(const float* a, const float* b, float* c,
                         size_t n, int inner) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float  x = a[idx];
    float  y = b[idx];
    float  acc = 0.f;

    // 10 000 次 FMAD（可由 inner 調整）
    #pragma unroll 4
    for (int i = 0; i < inner; ++i)
        acc = fmaf(x, y, acc);      // acc = x*y + acc;

    c[idx] = acc;
}


// ---------------- 計時 ----------------
float time_kernel(int bs,
                  const float* d_a,
                  const float* d_b,
                  float*       d_c)
{
    int grid = (N + bs - 1) / bs;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; ++i)
        vecAdd<<<grid, bs>>>(d_a, d_b, d_c, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));          // 等 GPU 完成

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / REPEAT;                         // 回傳單次平均 ms
}
float time_kernel2(int bs,
                  const float* d_a, const float* d_b, float* d_c,
                  int inner = 10000)          // ← 新增參數
{
    int grid = (N + bs - 1) / bs;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; ++i)
        heavyFma<<<grid, bs>>>(d_a, d_b, d_c, N, inner);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / REPEAT;
}
// ---------------- Main ----------------
int main() {
    // 1. host 端準備資料
    std::vector<float> h_a(N, 1.0f);            // 全 1
    std::vector<float> h_b(N, 2.0f);            // 全 2
    std::vector<float> h_c(N);

    // 2. device 端記憶體
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = N * sizeof(float);
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // 3. 送資料上 GPU
    CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    printf("bs   time(ms)  BW(GB/s)\n");
    printf("------------------------\n");

    // 4. 掃描不同 blockSize
    for (int bs = 32; bs <= 1024; bs += 32) {
        float ms = time_kernel(bs, d_a, d_b, d_c);
        double gb_s = 3.0 * N * sizeof(float) / (ms * 1e6); // 3 個陣列傳輸量
        printf("%4d  %7.3f   %8.2f\n", bs, ms, gb_s);
    }
	printf("bs   time(ms)  GFLOP/s\n");
	printf("-------------------------\n");

	double flop_per_elem = 2.0 * INNER;           // multiply+add = 2 FLOP
	double total_flop    = flop_per_elem * N;

	for (int bs = 32; bs <= 1024; bs += 32) {
	    float  ms   = time_kernel2(bs, d_a, d_b, d_c, INNER);
	    double gfls = total_flop / (ms * 1e6);
	    printf("%4d  %8.3f   %8.1f\n", bs, ms, gfls);
	}
    // 5.（可選）把結果抓回來驗證
    CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    if (h_c[12345] != 3.0f)
    {
	    printf("hc: %f\n", h_c[12345]);
	  fprintf(stderr, "結果驗證失敗！\n");
    }
    // 6. 清理
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return 0;
}
