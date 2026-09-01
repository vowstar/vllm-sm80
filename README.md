# vllm-sm80

This vLLM fork adds and validates model paths for NVIDIA sm_80 GPUs. The
repository keeps the GLM, DeepSeek Vision, and Qwen experimental work on one
`main` branch.

Development and testing use five CMP 170HX 64 GB GPUs with pipeline
parallelism and driver 610.43.02. Other sm_80 GPUs such as A100 are not yet
tested.

## Status

| Model | Result |
| --- | --- |
| GLM-5.3-Flash | Production tested with NVFP4, PP5, MTP x3, 1M context, vision, fp8 KV cache, and prefix caching. |
| DeepSeek-V4-Flash-Vision-Exp | Source and immutable image candidate are complete. Full hardware acceptance is pending. |
| Qwen3.8-Flash-Next-FP8 | Tested with PP5. Plain decoding starts, but performance and stability are not production ready. MTP with PP5 fails. |

The Qwen result is a tested limitation. It is not a support claim.

## GLM-5.3-Flash

### Validated features

| Feature | State |
| --- | --- |
| NVFP4 W4A16 MoE | Works with Marlin. A Triton emulation backend remains available as a fallback. |
| MTP x3 speculative decoding | Works |
| Context length | 1,048,576 tokens works |
| FP8 latent KV cache | Works with e4m3fn storage |
| Prefix caching | Works with the fixes in this fork |
| Vision input | Works |
| Tool calls and `glm47` parser | Work |

Measured on five CMP 170HX GPUs with PP5 partition `11,9,9,9,7`, MTP x3,
and the Marlin backend:

| Test | Result |
| --- | --- |
| Decode, one stream with a 200K prefix | About 100 tok/s |
| Decode, 10 concurrent 200K prefixes | About 382 tok/s aggregate |
| Cold prefill, one 200K prefix | About 32 s |

FP8 KV cache and bfloat16 KV cache were also compared on the same system:

| Test | FP8 KV | bfloat16 KV |
| --- | --- | --- |
| KV pool | 6.67M tokens | 3.79M tokens |
| Decode, one stream with a 1M prefix | About 71 tok/s | About 77 tok/s |
| Cold prefill, one 1M prefix | About 305 s | About 173 s |
| Needle recall at 128K, 512K, and 1M | All pass | All pass |

### Main GLM changes

| Area | Change |
| --- | --- |
| Sparse MLA attention | Adds a Triton NoPE kernel for sm_80 and an indexer fallback. |
| FP8 KV stores | Adds software e4m3fn encoding for sm_80. |
| FP8 latent KV cache | Uses uint8 storage and in-kernel dequantization without sm_89 FP8 instructions. |
| NVFP4 MoE | Adds the W4A16 path, Marlin repack holdoff, and a fused Triton emulation fallback. |
| Prefix caching | Ports the relevant scheduler fixes and adds uncached-first allocation, one cached FIFO, transient headroom, and optional diagnostics. |
| Large KV pools | Fixes integer width in the sparse MLA path. |
| Mamba postprocess | Uses a blocking device-to-host copy for the accepted token count. |

### Example GLM launch

Build an image from this checkout, then adapt the GPU list and layer partition
to the target system.

```bash
docker run -d --name vllm --runtime=nvidia \
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

| Setting | Reason |
| --- | --- |
| `--block-size 256` | The indexer requires a multiple of 128. The sparse MLA backend requires a multiple of 64. |
| `VLLM_PP_LAYER_PARTITION` | The last rank also carries `lm_head` and the MTP draft layer. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `143360` is aligned to 16 Mamba pages with fp8 KV. Use `73728` with bfloat16 KV. |
| `VLLM_MARLIN_REPACK_HOLDOFF` | Avoids a load-time MMU fault observed with CMP 170HX and driver 610.43.02. Other systems can set it to `0`. |
| `--kv-cache-memory` | Prevents Mamba state copies from evicting every hashed checkpoint when the default pool is too small. |

Known GLM limitations:

| Item | Detail |
| --- | --- |
| KDA numerics | Some tests differ from the reference by about 7 percent. |
| KV capacity | Mamba state pages alias into larger blocks, so effective capacity is below the raw allocation. |
| Transient headroom | The allocator keeps 32 blocks without hashes for Mamba state copies. Set `VLLM_APC_HEADROOM_BLOCKS=0` to disable it. |
| FP8 KV prefill | Prefill is about 1.8 times slower than bfloat16 KV at tested shapes. Decode speed is similar. The larger cache reduces repeated prefill. |
| Checkpoint indexes | This fork ignores `input_scale` entries that are not used by the W4A16 path, so old and new indexes both load. |

## DeepSeek V4 Flash Vision

This fork adds an experimental path for
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on sm_80. The source candidate is
[`bc3da91c8`](https://github.com/vowstar/vllm-sm80/commit/bc3da91c83889ecb228da3dc86a4f145a1abb016).

| Check | State |
| --- | --- |
| Vision source implementation | Complete |
| Immutable candidate image | Built, not promoted |
| Full checkpoint load | Pending |
| PP4 boot on four CMP 170HX GPUs | Pending |
| Image understanding and OCR | Pending |
| 1,048,576 token YaRN context | Pending |
| DSpark and MTP | Pending live checks |
| KV offload | Pending live checks |
| Prefix cache hit, eviction, and refill | Pending live checks |
| Sustained workload | Pending |

Implemented Vision changes:

| Area | Change |
| --- | --- |
| Model support | Adds the Vision tower, multimodal processor, registration, and weight loading. |
| Image routing | Routes image tokens through the `bias_vl` MoE path. |
| PrefixLM | Adds multimodal placeholders and image-local PrefixLM attention. |
| Ampere | Adds sparse sliding attention for image tokens on sm_80. |
| Pipeline parallelism | Carries vision metadata across PP ranks. |
| Speculative decoding | Adds the DeepSeek V4 DSpark path for pipeline parallel execution. |
| Embedding safety | Guards image token lookups before text embedding access. |
| Tests | Adds model, multimodal, PrefixLM, router, and sm_80 coverage. |

Promotion requires image understanding, OCR, three 1M-context needle checks,
working speculative decoding, KV offload restore, prefix-cache eviction and
refill, and a 600 second workload without request failures, Xid, AER, or
container restarts.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next-FP8 was tested on five CMP 170HX GPUs. The tested image
used vLLM main, PR #53899, and four local compatibility patches.

| Test | Result |
| --- | --- |
| PP5 without MTP | Boots and serves requests |
| Single-stream decode | 57.3 tok/s |
| Prefill | About one third of DeepSeek-V4-Flash-0731 on the same hardware |
| Concurrency 8 | Scaling was nearly flat |
| Concurrency 32 | A worker died and the engine stopped |
| MTP with PP5 | Fails with a CUDA illegal memory access in the target-model speculative path |

Repeated illegal memory accesses can leave the GPUs unable to create new CUDA
contexts until the driver state is recovered. The current upstream Qwen path
does not implement pipeline-parallel MTP for this topology. For these reasons,
this repository does not provide a production launch recipe for Qwen yet.

## Attribution

| Work | Author |
| --- | --- |
| GLM-5.3-Flash support, vLLM PR #53906 | ZJY0516 |
| Mamba align boundary fix, vLLM PR #53479 | kamb-code |
| Scheduler speculative decode padding fix, vLLM PR #53962 | njhill |
| CMP 170HX method reference | allover326/deepseek-v4-cmp170hx |

## History and maintenance

The repository uses one active branch, `main`. Pre-consolidation tips are
preserved as `archive/*-20260901` tags. Runtime deployment files live outside
this source repository.

This is a personal production fork. When upstream vLLM provides equivalent
sm_80 support, this repository will link to the upstream implementation.
