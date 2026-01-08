# nano vllm
## background
- 先從 比較入門的 nano vllm 入手 在比較去看 `vllm` 可能會比較快上手
## problem 
- 先看 `class LLMEngine` 主要再做什麼？ 會做 四件事情
1. `tensor parallel`: 
   - 主要工作: 可以把 tensor 平行部屬在多台 machine 上
   - 為什麼? 單卡放不了 tensor 才需要. 可以把矩陣切成多份每份都個別在 一張顯示卡.
2. `ModelRunner`: 
   - 主要工作:
     - load `safetensor`
     - 分配 kv cache , 每層 attention 的 `k_cache/v_cache`
     - prefill/decode 的 input packing?  (prepare_prefill / prepare_decode )
     - 執行 forward, sampler.
   - why? load weight 跟cache manager 集中管理, 並能做 `warnup`, `cudagraph`, `kv chche 分配` 
  
3. `Tokenizer` from transformer 
4. `schedule?` 
    - 主要工作 
      - 收 request -> 決定 prefill or decode?  
      - 依照 `max_num_seqs`, `max_num_batched_token` 組出 batch
      - 與 `BlockManager` 交互做 `kvcache block allocate` / `append` / `deallocate`. 
     - 為什麼：LLM 推理不是一條序列在跑，而是多序列共享 KV cache；排程能提高吞吐、避免爆記憶體
###  Q & A 
- what's `torch.multiprocessing` , tp parallel ?
  - `event` 主要用來同步不同機器的data
  - `process`: 會把 modelRunner部屬在個別的 rank
```python3 
ctx = mp.get_context("spawn")
for i in range(1, tensor_parallel_size)
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, arg...)
    process.start() 
    self.ps.append(process)
    self.events.append(event)
```
- 權重切分在 `nanovllm/layers/linear.py`：
  - `ColumnParallel`：切 output 維度，各 rank 各算一塊
  - `RowParallel`：切 input 維度，算完後 all_reduce 合併
  - `QKVParallel`：把 Q/K/V 的拼接向量切成各 rank 的 shard
## todo 
- [ ] 關於上方 wight 切分還不是很懂 明天在看
- [ ] scheduler ? 
- [ ] kv cache manager? 