# day 19
# study flashatten repo 
- prerequire: understand fa4 paper. 
    - Conditional softmax rescaling that skips unnecessary rescaling operations.
    - Software emulated exponential functions using polynomial approximation on FMA units.
    - Reducing shared memory traffic. 2-cta mma mode. each CTA stages and loads half of operand B. 
    - Restructure the dQ step to halve the number of `atomic reductions`. 
    - Implement a deterministic excution mode with minimal performace overhead, enabling reproducible training for reinforcement learning app. 
    - Implement new cta scheduling stragegies and register allocation  and `railored blackwell resource constraints and large tile size`
- read `flash_attn_func` in `cute/interface.py`. 
    - learn how to use the api, so need to understand every parameter usage.
    - the table is from claude code, but there are some parameter i don't known how to use it. 
        - softcap: 
        - return_lse: 
        - num_splits: 
        - pack_gqa: 
        - learnable_sink: 
        - mask_mod: 
        - block sparsity (flexattention):
Attention Core                                                                                                                                                                                                   
   
  ┌───────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                          
  │   Parameter   │                                                                               Meaning                                                                               │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                          
  │ softmax_scale │ Multiplier for Q*K^T scores. Default None → uses 1/sqrt(head_dim). Set manually if you want a different scaling (e.g. for MLA where key dim != value dim).          │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                          
  │ causal        │ Causal (autoregressive) mask — token at position i can only attend to positions <= i. Used in decoder-only models (GPT, LLaMA).                                     │                          
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                          
  │ window_size   │ (left, right) — sliding window attention. Token i attends to [i-left, i+right]. None means unbounded. E.g. (512, 0) = causal with 512-token window (Mistral-style). │                          
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                          
  │ softcap       │ Caps attention scores via softcap * tanh(scores / softcap) before softmax. Prevents extreme logits. Used by Gemma-2 (softcap=50.0). 0.0 = disabled.                 │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                          
  │ return_lse    │ Return the log-sum-exp of attention scores log(sum(exp(scores))) per row. Needed for SplitKV combine, loss computation, or custom backward.                         │
  └───────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘                          
                                                            
  Performance / Execution                                                                                                                                                                                          
                                                            
  ┌───────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  
  │   Parameter   │                                                                                           Meaning                                                                                           │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ num_splits    │ SplitKV (FlashDecoding): splits the K/V sequence into num_splits chunks, computes partial attention in parallel, then combines. Boosts GPU utilization for long KV with short Q             │
  │               │ (inference). 1 = no split. 0 = auto-select.                                                                                                                                                 │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ pack_gqa      │ GQA optimization: folds multiple Q heads per KV head into the sequence dimension, turning GQA into a bigger MHA problem. None = auto-enable when num_heads_q > num_heads_kv. Saves memory   │
  │               │ bandwidth.                                                                                                                                                                                  │  
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ deterministic │ Forces deterministic dQ accumulation in backward (uses atomics with fixed ordering). Slower but bitwise reproducible across runs.                                                           │  
  └───────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  
   
  Attention Modifiers                                                                                                                                                                                              
                                                            
  ┌────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  
  │   Parameter    │                                                                                          Meaning                                                                                           │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ learnable_sink │ Tensor of shape (num_heads,) — per-head learnable bias added to the first token's attention score. Implements "attention sinks" (StreamingLLM) where models dump attention mass on the     │
  │                │ first token.                                                                                                                                                                               │
  ├────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ mask_mod       │ A @cute.jit callable (b, h, q_idx, kv_idx) -> bool injected at compile time. Defines arbitrary attention patterns (document masking, block-diagonal, etc.). Mutually exclusive with        │
  │                │ causal.                                                                                                                                                                                    │  
  └────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                                                                                                   
  Block Sparsity (FlexAttention-style)                                                                                                                                                                             
   
  These 5 parameters work together — precomputed sparsity pattern telling the kernel which KV blocks to skip entirely:                                                                                             
                                                            
  ┌────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                                                     
  │   Parameter    │                                                                 Meaning                                                                 │
  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                     
  │ full_block_cnt │ (batch, heads, num_q_blocks) int32 — number of fully unmasked KV blocks per Q block. The kernel skips the softmax mask check for these. │
  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ full_block_idx │ (batch, heads, num_q_blocks, max_full) int32 — indices of those fully unmasked KV blocks.                                               │                                                     
  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                     
  │ mask_block_cnt │ (batch, heads, num_q_blocks) int32 — number of partially masked KV blocks per Q block. These need per-element mask evaluation.          │                                                     
  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                     
  │ mask_block_idx │ (batch, heads, num_q_blocks, max_mask) int32 — indices of those partially masked blocks.                                                │
  ├────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                     
  │ block_size     │ (tile_m, tile_n) — tile dimensions the sparsity was computed at. Must match the kernel's tile size.                                     │
  └────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘                                                     
                                                            

