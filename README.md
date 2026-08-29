# vllm-sm80

A vLLM fork that runs GLM-5.3-Flash on sm_80 GPUs. The base is vllm-project/vllm at PR #53906 (commit 142062f). Our work sits on top in the branch glm53-sm80.

We develop and test on 5x CMP 170HX 64 GB (GA100) with pipeline parallel 5, driver 610.43.02. Other sm_80 cards such as A100 are untested.

## Status

| Feature | State |
| --- | --- |
| GLM-5.3-Flash NVFP4 W4A16 MoE | Works. Marlin backend in production. Triton emulation backend as fallback. |
| MTP x3 speculative decoding | Works |
| Context 1,048,576 tokens | Works. KV pool 3.79M tokens across 5 ranks. |
| Prefix caching | Works after the fixes listed below |
| Vision input | Works |
| Tool calls, glm47 parser | Works |

Measured on 2026-08-28 and 2026-08-29, 5x CMP 170HX, PP5 partition 11,9,9,9,7, MTP x3, marlin backend:

| Test | Result |
| --- | --- |
| Decode, single stream, 200K prefix | about 100 tok/s |
| Decode, 10 concurrent 200K prefixes | about 382 tok/s aggregate |
| Cold prefill, one 200K prefix | about 32 s |

## Changes on top of #53906

| Area | Change |
| --- | --- |
| Sparse MLA attention | Triton kernel port for sm_80. NoPE path. Indexer Triton fallback. |
| FP8 KV stores | Software e4m3fn encoding on sm_80 |
| NVFP4 MoE | W4A16 path. Marlin repack holdoff. Fused Triton emulation kernel. |
| Prefix caching | Port of #53479. Port of #53962. Tiered free queue eviction policy with a bounded hit-proven tier. Env gated diagnostics (APCDIAG). |
| Sparse MLA kernel | int64 fix for KV pools above 4,194,304 rows |
| Mamba align postprocess | Blocking D2H copy for the accepted count |

## Run

This is our production shape. Adapt the GPU count and the layer partition to your cards.

```bash
docker run -d --name glm53 --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e VLLM_PP_LAYER_PARTITION=11,9,9,9,7 \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=73728 \
  -e VLLM_MARLIN_REPACK_HOLDOFF=1 \
  -v /path/to/GLM-5.3-Flash-NVFP4:/model \
  --shm-size=16g -p 8099:8000 \
  vllm-sm80:latest vllm serve /model \
  --served-model-name GLM-5.3-Flash \
  --pipeline-parallel-size 5 --kv-cache-dtype bfloat16 \
  --block-size 256 --max-model-len 1048576 \
  --max-num-batched-tokens 8192 --trust-remote-code \
  --gpu-memory-utilization 0.89 --max-num-seqs 32 \
  --kv-cache-memory 13421772800 \
  --reasoning-parser glm47 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --moe-backend marlin \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

| Flag | Why |
| --- | --- |
| `--block-size 256` | The indexer cache needs block_size % 128 == 0. The Triton sparse MLA backend declares MultipleOf(64). 256 satisfies both. |
| `VLLM_PP_LAYER_PARTITION` | The last rank also carries lm_head and the MTP draft layer. 11,9,9,9,7 is the most even split by checkpoint bytes. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | In scheduler tokens. 73728 is 16 mamba aligned pages of 4608 tokens. |
| `VLLM_MARLIN_REPACK_HOLDOFF` | Works around a load time MMU fault on CMP 170HX plus driver 610.43.02. Costs a few GiB transient during load. On other hardware, set it to 0. |
| `--kv-cache-memory` | The pool derived from gpu-memory-utilization is too small. At saturation the mamba state copies evict all hashed checkpoints. |

## Caveats

| Item | Detail |
| --- | --- |
| KDA numerics | The KDA path deviates about 7 percent from the reference in some tests |
| KV capacity | Mamba state pages alias into larger blocks. About 10 MB per block stays idle. Effective capacity is lower than the raw pool size suggests. |
| Hot tier cap | The hit-proven eviction tier demotes its oldest entries above 50 percent of the pool. Set VLLM_APC_HOT_CAP_PCT=0 for the unbounded behavior. |

## Attribution

| Work | By |
| --- | --- |
| GLM-5.3-Flash support in vLLM, PR #53906 | ZJY0516 |
| Mamba align boundary fix, PR #53479 | kamb-code |
| Scheduler spec decode padding fix, PR #53962 | njhill |
| CMP 170HX method reference | allover326/deepseek-v4-cmp170hx |

## Maintenance

This is a personal production fork. When upstream vLLM merges equivalent sm_80 support, this repository will link to it.
