# day3
## 目標
- 理解 hw spec 跟 搞懂 每個 block, thread, grid 才能更好的榨乾gpu 的性能. e.g., 資料怎麼安排
- hw: rtx 1070 8gb (compute capability 6.1)

| 指標                          | RTX 1070 (GP104)        |
| --------------------------- | ----------------------- |
| Compute Capability          | 6.1                     |
| SM 數                        | 15                      |
| 單 SM 最大常駐 thread            | 2048                    |
| 單 SM 可同時掛載 block            | ≤ 32                    |
| Warp size                   | 32                      |
| Max block size (1D)         | 1024 threads            |
| 每 SM Registers / Shared Mem | 65 536 x 32-bit / 64 KB |


- `sm` 同一時間可以跑 2048 threads, 但是每個 block 只能有 1024 threads, 所以同一時間只能有 兩個 1024 threads 的 block 在同一個 SM 上執行
- 在 nvidia gpu 每個warp 就是 32 threads, 所以每個 SM 同時執行的 warp 數量是 2048 / 32 = 64 warps. 
- warp schedule: 在 gpu 每次都是 32 threads 一起執行, warp schedule 就會根據每個 warp 的狀態來決定下一個要執行的 warp, 這樣可以避免因為某個 warp 在等待資料而導致其他 warp 無法執行的情況. 
- ref (https://modal.com/gpu-glossary/device-hardware/warp-scheduler)
## 今日遇到的問題
- computer capability version 需要注意 如果沒有指定 `-arch=sm_61` 可能會導致編譯錯誤, 或是執行結果 不正確 `nvcc -O3 -arch=sm_61 vecadd_bench.cu -o vecadd_bench`
> 執行 ncu 會有下面的 error 還要在看原因 `ncu --set launchstats --kernel-name-base vecAdd ./vectoradd
==ERROR== the argument for option '--kernel-name-base' is invalid. Use --help for further details.`
- 還有在看一下怎麼用 nv 的 tool 去做 profiling, 例如 `ncu` 或是 `nsight compute`, e.g., 今天跑 `ncu` 會有問題 明天在看一下要怎麼解決
## 進度: 比較 個別 block, thread, grid 下的 速度差異
- 在cuda 可以用下面的 api 來去 測量 時間, 在前後個別塞入`cudaEvent` 相關的 api 紀錄
```cpp
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

```
- 用下面的方式 寫 一個 kernel, 來比較 block, thread 的速度差異 
```cpp
for (int bs = 32; bs <= 1024; bs += 32) {
        float ms = time_kernel(bs, d_a, d_b, d_c);
        double gb_s = 3.0 * N * sizeof(float) / (ms * 1e6); // 3 個陣列傳輸量
        printf("%4d  %7.3f   %8.2f\n", bs, ms, gb_s);
    }
// kernel 
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


``` 

- 結果 : 理論上 block size 越大, 效率越高, 但是可能是我寫的 kernel 沒有達到 computer bound 的狀態, 所以 block size 跟 performance 呈現水平狀態. 
```
VecAdd  (memory-bound)
bs   time(ms)  BW(GB/s)
------------------------
  32     0.973     206.87
  64     0.967     208.25
  96     0.970     207.64
 128     0.970     207.62
 160     0.970     207.57
 192     0.970     207.61
 224     0.970     207.57
 256     0.970     207.49
 288     0.970     207.58
 320     0.970     207.58
 352     0.970     207.57
 384     0.970     207.57
 416     0.970     207.54
 448     0.970     207.55
 480     0.970     207.54
 512     0.970     207.46
 544     0.970     207.53
 576     0.970     207.49
 608     0.970     207.53
 640     0.970     207.59
 672     0.970     207.51
 704     0.971     207.32
 736     0.971     207.33
 768     0.971     207.33
 800     0.971     207.39
 832     0.970     207.45
 864     0.970     207.47
 896     0.970     207.48
 928     0.970     207.50
 960     0.970     207.50
 992     0.970     207.48
1024     0.971     207.44

RegBomb (compute-bound, 132 reg/thread)
bs   time(ms)  GFLOP/s
-------------------------
  32   638.803     8404.3
  64   642.392     8357.4
  96   646.571     8303.4
 128   646.527     8303.9
 160   646.440     8305.0
 192   646.263     8307.3
 224   645.992     8310.8
 256   645.903     8311.9
 288   645.828     8312.9
 320   645.821     8313.0
 352   646.894     8299.2
 384   648.425     8279.6
 416   648.193     8282.6
 448   647.907     8286.2
 480   647.658     8289.4
 512   647.271     8294.4
 544   646.915     8298.9
 576   646.711     8301.6
 608   646.404     8305.5
 640   646.245     8307.6
 672   646.008     8310.6
 704   648.842     8274.3
 736   644.935     8324.4
 768   644.922     8324.6
 800   670.901     8002.2
 832   645.537     8316.7
 864   645.339     8319.2
 896   644.916     8324.7
 928   667.723     8040.3
 960   645.906     8311.9
 992   645.788     8313.4
1024   645.393     8318.5
```

## 明天進度
- [ ] 看可不可以找到 compute bound 的 kernel, 來測試 block size 跟 performance 的關係.  


## ref 
https://ajdillhoff.github.io/notes/multidimensional_grids_and_data/?utm_source=chatgpt.com
https://forums.developer.nvidia.com/t/filter-profiled-kernels-with-nsight-compute-attach-mode/263714/2