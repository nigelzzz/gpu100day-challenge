

# TMA (tensor memory accelerate)

- hopper and blackwall. 

- 透過硬體去搬運記憶體

- gmem <→ smem

- blackwell 增加 boradcast ( 在多個 cta 共享的 分布式 smem

# UMMA

- blackwall , ptx \``tcgen05.mma`(umma)

- warpgroup MMA (WGMMA) 是 hopper. 允許 128 thread 跑同一條 mma

- blackwell umma 新加入了 tensor memory. 寫進去 tensor memory 而不是 register. ( 256KB( 

   - 優點 原先需要大量register. 

# warp specification
(https://zhuanlan.zhihu.com/p/1929932276499722808)
- hoppe and blackwall 

- 每個 warp 有個別專職的事情

more pipeline, more share mem, using mbarriar to sync.. 



- using async barriers…. 

![image.png](./A%20wonderful%20new%20card-assets/image.png)





# pipeline

- mma_arive (mma_mbar)

- mma_mbar.wait() 

- 由於 mma 結果寫在 TMEM, 各 warp 無須立即擁有 結果數據的register. 需要結果 透過\``tcgen05_ld` 

- tmem → high speed buffer, all warp can use….

- **warp 專職與數量配置**\
   FA4 在一個 CTA 內通常配置 **1×Load、1×MMA、8×Softmax、4×Correction、1–2×Epilogue** 這幾類 warp（合計形成多個 warpgroups），以追求各階段吞吐量平衡。實際 mapping 由 **TileScheduler** 決定；高效設定常用 **StaticPersistentTileScheduler**（每個 SM 常駐最多一個 CTA，以減少 launch overhead 並細緻重疊不同階段）

```cpp
// 啟動TMA載入K，指定完成後遞增mbarrier計數
cfk::copy_nobar(tKgK, tKsK, tmaLoadK, tma_load_mbar[0]);  
// ... 計算期間 ...  
// 在執行GEMM之前等待K的載入屏障達標
cfk::gemm_bar_wait(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0]);
```

- correction warp ( flashatten git repo

![image 1.png](./A%20wonderful%20new%20card-assets/image%201.png)

# code 

```cpp
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"

#include "kernel/fmha_tile_scheduler.hpp"

// main loop, epilogue  , schedule. 
```

- code 進去點 \`fwd runner\`

```cpp

struct FwdRunner {
...
using Operation = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShapeType,
      Mainloop,
      cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE
      >,
      TileScheduler
    >>;,.....
```