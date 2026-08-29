# vllm-sm80

A vLLM fork that runs GLM-5.3-Flash on sm_80 GPUs. The base is vllm-project/vllm at PR #53906 (commit 142062f). Our work sits on top in the branch glm53-sm80.

We develop and test on 5x CMP 170HX 64 GB (GA100) with pipeline parallel 5, driver 610.43.02. Other sm_80 cards such as A100 are untested.

## Status

| Feature | State |
| --- | --- |
| GLM-5.3-Flash NVFP4 W4A16 MoE | Works. Marlin backend in production. Triton emulation backend as fallback. |
| MTP x3 speculative decoding | Works |
| Context 1,048,576 tokens | Works. KV pool 6.67M tokens across 5 ranks with fp8 KV cache. |
| FP8 latent KV cache | Works. e4m3fn, default since 2026-08-30. bfloat16 stays selectable. |
| Prefix caching | Works after the fixes listed below |
| Vision input | Works |
| Tool calls, glm47 parser | Works |

Measured on 2026-08-28 and 2026-08-29, 5x CMP 170HX, PP5 partition 11,9,9,9,7, MTP x3, marlin backend:

| Test | Result |
| --- | --- |
| Decode, single stream, 200K prefix | about 100 tok/s |
| Decode, 10 concurrent 200K prefixes | about 382 tok/s aggregate |
| Cold prefill, one 200K prefix | about 32 s |

Measured on 2026-08-30, same shape, fp8 KV cache against bfloat16 KV cache:

| Test | fp8 KV | bfloat16 KV |
| --- | --- | --- |
| KV pool | 6.67M tokens | 3.79M tokens |
| Decode, single stream, 1M prefix | about 71 tok/s | about 77 tok/s |
| Cold prefill, one 1M prefix | about 305 s | about 173 s |
| Needle recall at 128K, 512K, 1M | all pass | all pass |

## Changes on top of #53906

| Area | Change |
| --- | --- |
| Sparse MLA attention | Triton kernel port for sm_80. NoPE path. Indexer Triton fallback. |
| FP8 KV stores | Software e4m3fn encoding on sm_80 |
| FP8 latent KV cache | e4m3fn with per tensor scale. uint8 storage plus a 6 op bit assembly dequant to fp16 inside the Triton sparse MLA kernel, so no sm_89 fp8 support is needed. Pool grows from 3.79M to 6.67M tokens. |
| NVFP4 MoE | W4A16 path. Marlin repack holdoff. Fused Triton emulation kernel. |
| Prefix caching | Port of #53479. Port of #53962. Uncached-first allocation with a single cached FIFO and un-hashed headroom. Env gated diagnostics (APCDIAG). |
| Sparse MLA kernel | int64 fix for KV pools above 4,194,304 rows |
| Mamba align postprocess | Blocking D2H copy for the accepted count |

## Run

This is our production shape. Adapt the GPU count and the layer partition to your cards.

```bash
docker run -d --name glm53 --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e VLLM_PP_LAYER_PARTITION=11,9,9,9,7 \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=143360 \
  -e VLLM_MARLIN_REPACK_HOLDOFF=1 \
  -v /path/to/GLM-5.3-Flash-NVFP4:/model \
  --shm-size=16g -p 8099:8000 \
  vllm-sm80:latest vllm serve /model \
  --served-model-name GLM-5.3-Flash \
  --pipeline-parallel-size 5 --kv-cache-dtype fp8 \
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
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | In scheduler tokens. 143360 is 16 mamba aligned pages of 8960 tokens with fp8 KV. With bfloat16 KV the page is 4608 tokens, use 73728. The scheduler rejects values that are not multiples of the page size. |
| `VLLM_MARLIN_REPACK_HOLDOFF` | Works around a load time MMU fault on CMP 170HX plus driver 610.43.02. Costs a few GiB transient during load. On other hardware, set it to 0. |
| `--kv-cache-memory` | The pool derived from gpu-memory-utilization is too small. At saturation the mamba state copies evict all hashed checkpoints. |

## Caveats

| Item | Detail |
| --- | --- |
| KDA numerics | The KDA path deviates about 7 percent from the reference in some tests |
| KV capacity | Mamba state pages alias into larger blocks. About 10 MB per block stays idle. Effective capacity is lower than the raw pool size suggests. |
| Headroom | 32 blocks stay un-hashed for transient mamba state allocations. Set VLLM_APC_HEADROOM_BLOCKS=0 to disable. |
| FP8 KV prefill cost | Prefill runs about 1.8 times slower than bfloat16 KV because the dequant is ALU bound at prefill shapes. Decode speed is unchanged. The larger prefix cache pool offsets repeat prefill work. |

## Attribution

| Work | By |
| --- | --- |
| GLM-5.3-Flash support in vLLM, PR #53906 | ZJY0516 |
| Mamba align boundary fix, PR #53479 | kamb-code |
| Scheduler spec decode padding fix, PR #53962 | njhill |
| CMP 170HX method reference | allover326/deepseek-v4-cmp170hx |

## Maintenance

This is a personal production fork. When upstream vLLM merges equivalent sm_80 support, this repository will link to it.
