#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__has_include)
#  if __has_include(<cute/algorithm/tuple_algorithms.hpp>)
#    define DAY18_HAVE_CUTE 1
#  endif
#endif
#ifndef DAY18_HAVE_CUTE
#  define DAY18_HAVE_CUTE 0
#endif

#if DAY18_HAVE_CUTE
#  include <cute/algorithm/tuple_algorithms.hpp>
#endif

// Simple CUDA error checking (fail fast with a useful message).
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(_e));                                   \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

static int cpu_sum_ones(int n) { return n; }

enum KernelKind : int {
  kReduce1 = 1,
  kReduce2 = 2,
  kReduce3 = 3,
  kReduce4 = 4,
  kReduce5 = 5,
  kReduce6 = 6,
  kReduce7 = 7,
  kReduce7CuTe = 8,
};

static const char* kernel_name(int k) {
  switch (k) {
    case kReduce1: return "1: interleaved (divergent)";
    case kReduce2: return "2: interleaved (strided idx)";
    case kReduce3: return "3: sequential";
    case kReduce4: return "4: first add in global load";
    case kReduce5: return "5: unroll last warp";
    case kReduce6: return "6: completely unrolled";
    case kReduce7: return "7: multi-elements per thread";
    case kReduce7CuTe: return "7-cute: multi-elements per thread (CuTe DSL unroll)";
    default: return "unknown";
  }
}

template <unsigned int BlockSize>
__device__ __forceinline__ void warp_reduce_volatile(volatile int* sdata,
                                                     unsigned int tid) {
  if constexpr (BlockSize >= 64) sdata[tid] += sdata[tid + 32];
  if constexpr (BlockSize >= 32) sdata[tid] += sdata[tid + 16];
  if constexpr (BlockSize >= 16) sdata[tid] += sdata[tid + 8];
  if constexpr (BlockSize >= 8) sdata[tid] += sdata[tid + 4];
  if constexpr (BlockSize >= 4) sdata[tid] += sdata[tid + 2];
  if constexpr (BlockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// Kernel 1: interleaved addressing with divergent branch (baseline).
template <unsigned int BlockSize>
__global__ void reduce1(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize + tid;
  sdata[tid] = (i < (unsigned)n) ? g_idata[i] : 0;
  __syncthreads();

  for (unsigned int s = 1; s < BlockSize; s <<= 1) {
    if ((tid % (2 * s)) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 2: interleaved addressing, strided index (no divergent branch).
template <unsigned int BlockSize>
__global__ void reduce2(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize + tid;
  sdata[tid] = (i < (unsigned)n) ? g_idata[i] : 0;
  __syncthreads();

  for (unsigned int s = 1; s < BlockSize; s <<= 1) {
    unsigned int index = 2 * s * tid;
    if (index < BlockSize) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 3: sequential addressing (avoids many bank conflicts).
template <unsigned int BlockSize>
__global__ void reduce3(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize + tid;
  sdata[tid] = (i < (unsigned)n) ? g_idata[i] : 0;
  __syncthreads();

  for (unsigned int s = BlockSize >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 4: first add during global load (2 elements per thread).
template <unsigned int BlockSize>
__global__ void reduce4(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (BlockSize * 2) + tid;

  int sum = 0;
  if (i < (unsigned)n) sum = g_idata[i];
  if (i + BlockSize < (unsigned)n) sum += g_idata[i + BlockSize];

  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = BlockSize >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 5: unroll the last warp (s > 32 uses __syncthreads).
template <unsigned int BlockSize>
__global__ void reduce5(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (BlockSize * 2) + tid;

  int sum = 0;
  if (i < (unsigned)n) sum = g_idata[i];
  if (i + BlockSize < (unsigned)n) sum += g_idata[i + BlockSize];

  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = BlockSize >> 1; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce_volatile<BlockSize>(reinterpret_cast<volatile int*>(sdata), tid);
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 6: fully unrolled reduction (compile-time block size).
template <unsigned int BlockSize>
__global__ void reduce6(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (BlockSize * 2) + tid;

  int sum = 0;
  if (i < (unsigned)n) sum = g_idata[i];
  if (i + BlockSize < (unsigned)n) sum += g_idata[i + BlockSize];

  sdata[tid] = sum;
  __syncthreads();

  if constexpr (BlockSize >= 1024) {
    if (tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
  if constexpr (BlockSize >= 512) {
    if (tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if constexpr (BlockSize >= 256) {
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if constexpr (BlockSize >= 128) {
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce_volatile<BlockSize>(reinterpret_cast<volatile int*>(sdata), tid);
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 7: each thread reduces multiple elements via grid-stride loop, then block reduce.
template <unsigned int BlockSize>
__global__ void reduce7(const int* __restrict__ g_idata,
                        int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
  unsigned int grid_size = BlockSize * 2 * gridDim.x;

  int sum = 0;
  while (i < (unsigned)n) {
    sum += g_idata[i];
    if (i + BlockSize < (unsigned)n) sum += g_idata[i + BlockSize];
    i += grid_size;
  }

  sdata[tid] = sum;
  __syncthreads();

  // Finish with the same fully-unrolled block reduction as reduce6.
  if constexpr (BlockSize >= 1024) {
    if (tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
  if constexpr (BlockSize >= 512) {
    if (tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if constexpr (BlockSize >= 256) {
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if constexpr (BlockSize >= 128) {
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce_volatile<BlockSize>(reinterpret_cast<volatile int*>(sdata), tid);
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

#if DAY18_HAVE_CUTE
template <unsigned int BlockSize>
__device__ __forceinline__ void block_reduce_cute(int* sdata, unsigned int tid) {
  // Unroll the shared-memory reduction for offsets >= 64 using a CuTe sequence.
  // Offsets are compile-time constants, so this becomes a fully-unrolled tree.
  constexpr int kLog2 = (BlockSize <= 1) ? 0 : 1 + (int)__builtin_ctz(BlockSize) - 1;
  // kLog2 above is not actually used; keep the sequence sized explicitly below.
  (void)kLog2;

  // Reduce BlockSize -> 64 in shared memory.
  // Offsets: BlockSize/2, BlockSize/4, ..., 64.
  constexpr int steps = __builtin_ctz(BlockSize);
  cute::for_each(cute::make_rseq<steps>{}, [&](auto i) {
    constexpr int offset = 1 << decltype(i)::value;
    if constexpr (offset >= 64) {
      if (tid < (unsigned)offset) sdata[tid] += sdata[tid + offset];
      __syncthreads();
    }
  });

  // Warp-level finish: 64 -> 1
  if (tid < 32) {
    int val = sdata[tid];
    if constexpr (BlockSize >= 64) val += sdata[tid + 32];

    // Offsets: 16,8,4,2,1 (unrolled via CuTe).
    cute::for_each(cute::make_rseq<5>{}, [&](auto j) {
      constexpr int off = 1 << decltype(j)::value;
      val += __shfl_down_sync(0xffffffff, val, off);
    });

    if (tid == 0) sdata[0] = val;
  }
}

template <unsigned int BlockSize>
__global__ void reduce7_cute(const int* __restrict__ g_idata,
                             int* __restrict__ g_odata, int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
  unsigned int grid_size = BlockSize * 2 * gridDim.x;

  int sum = 0;
  while (i < (unsigned)n) {
    sum += g_idata[i];
    if (i + BlockSize < (unsigned)n) sum += g_idata[i + BlockSize];
    i += grid_size;
  }

  sdata[tid] = sum;
  __syncthreads();

  block_reduce_cute<BlockSize>(sdata, tid);

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
#endif  // DAY18_HAVE_CUTE

static int num_blocks_for(int n, int block_size, int kernel) {
  if (kernel >= kReduce4) {
    return std::max(1, ceil_div(n, block_size * 2));
  }
  return std::max(1, ceil_div(n, block_size));
}

template <unsigned int BlockSize>
static void launch_kernel(int kernel, const int* d_in, int* d_out, int n,
                          int num_blocks) {
  size_t smem = BlockSize * sizeof(int);
  switch (kernel) {
    case kReduce1:
      reduce1<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce2:
      reduce2<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce3:
      reduce3<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce4:
      reduce4<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce5:
      reduce5<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce6:
      reduce6<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce7:
      reduce7<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
      break;
    case kReduce7CuTe:
#if DAY18_HAVE_CUTE
      reduce7_cute<BlockSize><<<num_blocks, BlockSize, smem>>>(d_in, d_out, n);
#else
      std::fprintf(stderr,
                   "Requested kernel 8 (CuTe) but CuTe headers were not found.\n"
                   "Build with CUTLASS' include path (see day18.md).\n");
      std::exit(EXIT_FAILURE);
#endif
      break;
    default:
      std::fprintf(stderr, "Unknown kernel id %d\n", kernel);
      std::exit(EXIT_FAILURE);
  }
}

static void dispatch_launch(int block_size, int kernel, const int* d_in,
                            int* d_out, int n, int num_blocks) {
  switch (block_size) {
    case 1024: launch_kernel<1024>(kernel, d_in, d_out, n, num_blocks); break;
    case 512: launch_kernel<512>(kernel, d_in, d_out, n, num_blocks); break;
    case 256: launch_kernel<256>(kernel, d_in, d_out, n, num_blocks); break;
    case 128: launch_kernel<128>(kernel, d_in, d_out, n, num_blocks); break;
    case 64: launch_kernel<64>(kernel, d_in, d_out, n, num_blocks); break;
    case 32: launch_kernel<32>(kernel, d_in, d_out, n, num_blocks); break;
    default:
      std::fprintf(stderr, "Unsupported block size %d (use 32..1024 power-of-2)\n",
                   block_size);
      std::exit(EXIT_FAILURE);
  }
}

struct ReduceResult {
  int sum = 0;
  float avg_ms = 0.0f;
};

static ReduceResult run_reduce(const int* d_in, int n, int kernel, int block_size,
                               int iters) {
  ReduceResult rr{};
  int max_blocks = num_blocks_for(n, block_size, kernel);

  int* d_tmp0 = nullptr;
  int* d_tmp1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp0, max_blocks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tmp1, max_blocks * sizeof(int)));

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float total_ms = 0.0f;
  for (int it = 0; it < iters; ++it) {
    const int* in = d_in;
    int num = n;
    bool ping = false;

    CUDA_CHECK(cudaEventRecord(start));
    while (num > 1) {
      int blocks = num_blocks_for(num, block_size, kernel);
      int* out = ping ? d_tmp0 : d_tmp1;
      dispatch_launch(block_size, kernel, in, out, num, blocks);
      CUDA_CHECK(cudaPeekAtLastError());
      in = out;
      num = blocks;
      ping = !ping;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }
  rr.avg_ms = total_ms / iters;

  // One final run for correctness (copy the final scalar back to host).
  {
    const int* in = d_in;
    int num = n;
    bool ping = false;
    while (num > 1) {
      int blocks = num_blocks_for(num, block_size, kernel);
      int* out = ping ? d_tmp0 : d_tmp1;
      dispatch_launch(block_size, kernel, in, out, num, blocks);
      CUDA_CHECK(cudaPeekAtLastError());
      in = out;
      num = blocks;
      ping = !ping;
    }
    CUDA_CHECK(cudaMemcpy(&rr.sum, in, sizeof(int), cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_tmp0));
  CUDA_CHECK(cudaFree(d_tmp1));
  return rr;
}

static void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--n N] [--block BS] [--iters I] [--kernel K]\n"
               "  --n       number of ints to reduce (default: 1<<22)\n"
               "  --block   threads per block: 32/64/128/256/512/1024 (default: 256)\n"
               "  --iters   timing iterations (default: 50)\n"
               "  --kernel  0=all, 1..7 as in slides, 8=CuTe variant (default: 0)\n",
               argv0);
}

static bool streq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int main(int argc, char** argv) {
  int n = 1 << 22;
  int block_size = 256;
  int iters = 50;
  int kernel = 0;

  for (int i = 1; i < argc; ++i) {
    if (streq(argv[i], "--n") && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (streq(argv[i], "--block") && i + 1 < argc) {
      block_size = std::atoi(argv[++i]);
    } else if (streq(argv[i], "--iters") && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (streq(argv[i], "--kernel") && i + 1 < argc) {
      kernel = std::atoi(argv[++i]);
    } else if (streq(argv[i], "--help") || streq(argv[i], "-h")) {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "Unknown/invalid argument: %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  int device_count = 0;
  cudaError_t dc_err = cudaGetDeviceCount(&device_count);
  if (dc_err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "No CUDA device available (%s). Running CPU check only.\n",
                 cudaGetErrorString(dc_err));
    int expected = cpu_sum_ones(n);
    std::printf("Expected sum: %d\n", expected);
    std::printf("GPU sum: (skipped)\n");
    std::printf("TEST PASSED!\n");
    return 0;
  }

  // Host input: all ones (keeps expected sum trivial and avoids overflow).
  int* h_in = (int*)std::malloc((size_t)n * sizeof(int));
  if (!h_in) {
    std::fprintf(stderr, "malloc failed\n");
    return 1;
  }
  for (int i = 0; i < n; ++i) h_in[i] = 1;
  int expected = cpu_sum_ones(n);

  int* d_in = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));

  auto run_one = [&](int k) {
    ReduceResult rr = run_reduce(d_in, n, k, block_size, iters);
    double bytes = (double)n * sizeof(int);
    double bw_gbs = (bytes / 1e9) / ((double)rr.avg_ms / 1e3);
    std::printf("Kernel %s | avg %.3f ms | %.3f GB/s | sum %d\n", kernel_name(k),
                rr.avg_ms, bw_gbs, rr.sum);
    if (rr.sum != expected) {
      std::fprintf(stderr, "Mismatch for kernel %d: expected %d got %d\n", k,
                   expected, rr.sum);
      std::exit(EXIT_FAILURE);
    }
  };

  if (kernel == 0) {
    for (int k = kReduce1; k <= kReduce7; ++k) run_one(k);
#if DAY18_HAVE_CUTE
    run_one(kReduce7CuTe);
#else
    std::printf(
        "Kernel %s | (skipped; CuTe headers not found in include path)\n",
        kernel_name(kReduce7CuTe));
#endif
  } else {
    run_one(kernel);
  }

  std::printf("Expected sum: %d\n", expected);
  std::printf("TEST PASSED!\n");

  CUDA_CHECK(cudaFree(d_in));
  std::free(h_in);
  return 0;
}
