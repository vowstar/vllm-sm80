# vllm-sm80

This vLLM fork runs three large models on NVIDIA sm_80 GPUs that upstream vLLM
does not support there. All three share one `main` branch and one image.

Development and testing use CMP 170HX 64 GB cards with pipeline parallelism and
driver 610.43.02. Other sm_80 GPUs such as A100 are not tested.

## Status

| Model | State | Tested layout |
| --- | --- | --- |
| DeepSeek-V4-Flash-Vision-Exp | Serving, measured | 4 or 5 cards, PP4 or PP5, 1M context, vision, DSpark x3, fp8 KV |
| Qwen3.8-Flash-Next-FP8 | Serving, measured | 4 cards, PP4, 1M YaRN context, PLE CPU offload |
| GLM-5.3-Flash | Serving, measured | 5 cards, PP5, 1M context, vision, MTP x3, fp8 KV |

DeepSeek-V4-Flash-0731, the text only checkpoint, also loads and answers on
this branch. It was validated at 64K context without speculative decoding.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | sm_80. Tested only on CMP 170HX 64 GB, which has no peer to peer and a 64 MB BAR1. |
| Cards | 4 for Qwen, 5 for GLM, 4 or 5 for DeepSeek. Tensor parallelism does not work on these cards, so the count is a pipeline depth. |
| VRAM | 64 GB per card. The tightest rank runs within about 1.6 GiB of the limit. |
| Host RAM | DeepSeek costs about 11 GB once serving, but weight load peaks far higher, so leave 30 GB free. Qwen needs about 70 GB, because its PLE table is 51 GB and lives in host memory. |
| Disk | 156 to 170 GiB per checkpoint. |
| Driver | 610.43.02. Other drivers are untested and the Marlin holdoff below exists because of this one. |

## Building the image

Build with the upstream Dockerfile and restrict the architecture list, which
cuts compile time by a large factor:

```bash
docker build -f docker/Dockerfile --target vllm-openai \
  --build-arg torch_cuda_arch_list=8.0 \
  --build-arg max_jobs=8 \
  -t vllm-sm80:$(git rev-parse --short HEAD) .
```

That is the whole build. Nothing else in this repository is required.

## Measured performance

Two hosts, each five CMP 170HX 64 GB cards behind a shared PCIe uplink with no
peer to peer. All three columns come from one harness against a live service
with real technical prose as the prompt.

Use real prose. A prompt built from one repeated word makes speculative
decoding accept almost everything at short context and almost nothing at long
context, which turns a flat curve into a cliff. The same DeepSeek service
measured 101 tok/s at 2 K and 29 tok/s at 1 M on a repeated word prompt, and
76 tok/s and 26 tok/s on prose. Only the prose numbers mean anything.

### Decode speed against context length

One stream, cold prompt, no prefix cache hit. Time to first token is the full
prefill.

| Prompt tokens | DeepSeek tok/s | Qwen tok/s | GLM tok/s |
| ---: | ---: | ---: | ---: |
| About 2 K | 75.8 | 53.3 | 68.7 |
| About 7 K | 66.4 | 51.4 | 60.8 |
| About 30 K | 77.9 | 52.3 | 64.3 |
| About 90 K | 58.2 | 52.0 | 62.7 |
| About 230 K | 58.5 | 53.0 | 62.5 |
| About 460 K | 48.2 | 55.1 | 54.0 |
| About 900 K to 1 M | 26.0 | 54.1 | 68.3 |

Read the three columns as three attention designs, not as a ranking.

Qwen holds one decode rate at every length because its GDN linear attention
keeps a constant size recurrent state. GLM is also close to flat because only
11 of its 45 layers are sparse MLA and the other 34 are KDA linear attention.
DeepSeek is the only one that falls away, because its sparse indexer scores
every position in the cache on every step and its DSpark acceptance drops with
context.

The crossovers matter more than the peaks. DeepSeek leads GLM up to about
30 K and trails it after that. DeepSeek leads Qwen up to somewhere between
230 K and 460 K, and past that Qwen is both faster to decode and several times
faster to prefill.

Two caveats on this table. Only DeepSeek and GLM run speculative decoding, so
part of their short context advantage is draft acceptance rather than step
time. The GLM column uses fp8 KV, the production default, so it decodes a few
tokens per second slower than the same model with bfloat16 KV; the comparison
table below quantifies the gap.

### Prefill

Time to first token on the same runs, with the rate it implies.

| Prompt tokens | DeepSeek | Qwen | GLM |
| ---: | ---: | ---: | ---: |
| About 2 K | 0.9 s, 2,163 tok/s | 0.5 s, 4,102 tok/s | 1.2 s, 1,439 tok/s |
| About 30 K | 5.8 s, 5,225 tok/s | 2.9 s, 10,267 tok/s | 8.8 s, 3,089 tok/s |
| About 90 K | 17.9 s, 5,146 tok/s | 6.2 s, 14,499 tok/s | 20.9 s, 3,973 tok/s |
| About 230 K | 53.1 s, 4,336 tok/s | 15.3 s, 14,670 tok/s | 49.1 s, 4,216 tok/s |
| About 460 K | 138.0 s, 3,334 tok/s | 33.9 s, 13,277 tok/s | 102.1 s, 4,055 tok/s |
| About 900 K to 1 M | 406.7 s, 2,262 tok/s | 82.9 s, 10,751 tok/s | 289.0 s, 3,177 tok/s |

Qwen prefills two to four times faster than DeepSeek at every length and holds
its rate, while DeepSeek peaks near 30 K and then halves. GLM peaks near
4,200 tok/s around 230 K and drops to about 3,200 tok/s at 1 M. Its fp8 KV
prefill is slower than bfloat16 KV, as the comparison table below shows.

### Capacity and start up

Measured on the running services. The KV pool is what the engine reports after
it sizes the cache.

| | DeepSeek Vision | Qwen | GLM |
| --- | ---: | ---: | ---: |
| Checkpoint on disk | 156 GiB | 173 GiB | 183 GiB |
| Parameters | 305 B, about 16 B active | 180 B, 51 B of it the PLE table, about 10 B active | 320 B, 18 B active |
| Cards, pipeline stages | 4, PP4 | 4, PP4 | 5, PP5 |
| Weights per card | 41.2 GiB | About 55 GiB | About 49 to 64 GiB |
| KV pool | 2,655,371 tokens | 2,885,563 tokens | 6,670,108 tokens |
| Concurrency at 1 M context | 2.53x | 2.89x | 6.36x |
| Cold start to serving | 4 minutes from NVMe | 35 minutes from spinning disks | 4 minutes from NVMe |

The Qwen cold start is dominated by reading 173 GiB of weights, 51 GB of it
the PLE table, off spinning disks. The same model on NVMe would not take that long.
GLM holds a much larger KV pool because its 34 KDA layers store a fixed size
recurrent state instead of a growing cache, and because its launcher pins
`--kv-cache-memory` at 12.5 GiB per rank.

### Throughput against concurrency

The three models run different topologies (DeepSeek PP4 and PP5, Qwen PP4,
GLM PP5) and different true-concurrency caps (`--max-num-seqs`). A shared
throughput table is not published because those differences make one "streams"
count mean different things per model.

**Two throughput numbers exist and they are not interchangeable.** Quoting one
against the other is the most common way these figures get misread:

- **aggregate** is total output tokens divided by total wall clock. Time to
  first token, the pipeline fill and the drain at the end are all charged
  against it. It is the honest end-to-end number for a fixed batch of work.
- **steady generation** is the engine's own `Avg generation throughput` while
  the requested concurrency is actually resident. It is what a serving
  dashboard displays, and it excludes everything before the plateau.

On a four-stage pipeline the gap between them is large and shrinks as the run
gets longer, because a PP4 pipeline cannot reach steady state until enough
requests are in flight. Measured on Qwen PP4, 5x CMP 170HX, prose prompts of
about 1,600 tokens with unique prefixes so the prefix cache never hits:

| Concurrency | Output/req | Wall | aggregate | steady generation | per-request median |
| --- | --- | --- | --- | --- | --- |
| 1 | 512 | 22.6 s | 22.7 tok/s | 38.9 tok/s | 22.7 tok/s |
| 4 | 512 | 25.3 s | 80.9 tok/s | 49.4 tok/s | 20.2 tok/s |
| 8 | 512 | 19.7 s | 207.7 tok/s | 158.8 tok/s | 26.0 tok/s |
| 8 | 2048 | 64.6 s | **253.7 tok/s** | **259.1 tok/s** | 31.7 tok/s |
| 16 | 512 | 21.8 s | 375.5 tok/s | 432.5 tok/s | 23.5 tok/s |
| 32 | 512 | 32.4 s | 506.0 tok/s | 644.3 tok/s | 15.8 tok/s |

The last two rows are the same server and the same concurrency; only the
output length differs. With 512-token outputs the run is too short for the
pipeline to fill, so aggregate and steady disagree by a third. With
2048-token outputs they converge to within 2 percent. Any comparison that
puts one system's steady dashboard reading next to another system's
short-run aggregate is measuring output length, not the engine.

The engine's own plateau lines for the 2048-token run, for reference:

```
Avg generation throughput: 257.0 tokens/s, Running: 8 reqs, Prefix cache hit rate: 0.0%
Avg generation throughput: 265.7 tokens/s, Running: 8 reqs, Prefix cache hit rate: 0.0%
Avg generation throughput: 261.3 tokens/s, Running: 8 reqs, Prefix cache hit rate: 0.0%
Avg generation throughput: 265.9 tokens/s, Running: 8 reqs, Prefix cache hit rate: 0.0%
```

The QSA Triton kernels used to JIT on the first request. On a pipeline-parallel
rank that JIT lands inside a collective: one rank enters `cuModuleLoad` while
its peers spin in `recv`, and `--max-num-seqs 32` never finished starting. The
fix pre-compiles the kernels during warmup: un-gate the Qwen model type in the
Triton warmup, warm the four QSA kernels on their exact block-table widths and
every batch size the scheduler can produce, and mark the runtime integer
scalars `do_not_specialize` so they no longer recompile per shape. 16 and 32
concurrent streams now serve with zero first-request JIT (see the table above).

An earlier revision of this file claimed 32 streams at 804.8 tok/s aggregate
with no crash. That did not survive re-measurement and has been removed.

An earlier build killed the Qwen engine at 32 streams. The PLE offload request
queue held one entry and the producer used `put_nowait`, which assumes each
forward is consumed before the next launch. Pipeline parallelism runs the
engine batch queue and breaks that assumption, so the queue filled, the rank 0
worker raised `queue.Full`, and the engine died five minutes later on an RPC
timeout. The producer now blocks on a 60 s bound, and 32 streams is stable.

## DeepSeek V4 Flash and Vision

The Vision checkpoint is `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.

| Feature | State |
| --- | --- |
| 1,048,576 token context | Works |
| Image input | Works, checked against a generated test image |
| DSpark speculative decoding under PP | Works |
| Prefix caching | Works |
| Tool calls and the `deepseek_v4` parser | Works, checked with a function call |
| KV offload | Present, not exercised here |

Main changes for this model:

| Area | Change |
| --- | --- |
| Vision | Adds the vision tower, multimodal processor, registration, and weight loading. |
| Image routing | Routes image tokens through the `bias_vl` MoE path. |
| PrefixLM | Adds multimodal placeholders and image local PrefixLM attention. |
| Ampere | Adds sparse sliding attention for image tokens on sm_80. |
| Pipeline parallelism | Carries vision metadata across PP ranks and adds the DSpark PP path. |
| Prefix caching | Keeps the deepest EAGLE reachable boundary for sparse sliding window groups. |
| Metrics | Caps speculative acceptance at the number of drafts a grammar left valid. |

## Qwen3.8-Flash-Next

The checkpoint is `Qwen3.8-Flash-Next-FP8` with the `qwen4_exp` architecture.
Upstream [PR #53899](https://github.com/vllm-project/vllm/pull/53899) validated
tensor parallel and data parallel only, so the pipeline parallel path here is
this fork's work.

| Feature | State |
| --- | --- |
| PP4 serving | Works |
| 1,000,000 token YaRN context | Works, from a native 262,144 |
| PLE CPU offload under PP | Works |
| 8 concurrent streams | Works, 259 tok/s steady generation, cold prefix cache |
| 16 concurrent streams | Works, 467 tok/s steady generation |
| 32 concurrent streams | Does not finish starting, driver-level module-load stall |
| Prefix caching | Works |
| Vision tower | Loads and warms up, image accuracy not checked |
| MTP | Not available under PP |

The PLE table is a 51 GB FP8 ngram embedding of 16 heads over a 20 million
entry vocabulary, which is 51 of the model's 180 billion parameters. It stays in host memory and costs about microseconds per
token, so it is not the decode bottleneck. It does need roughly 63 GB of host
RAM for the whole container, and its staging thread is what fails first under
heavy concurrency.

Main changes for this model:

| Area | Change |
| --- | --- |
| PLE offload | Enables it on rank 0 under pipeline parallelism instead of rejecting PP. |
| PLE offload | Waits for the staging queue instead of raising `queue.Full` and killing the engine. |
| Model state | Disables ngram state on non first ranks rather than failing. |
| Weight loading | Skips the final mixer weights on non last ranks. |
| Marlin FP8 | Adds the repack holdoff that CMP 170HX needs. |

## GLM-5.3-Flash

| Feature | State |
| --- | --- |
| NVFP4 W4A16 MoE | Works with Marlin, with a Triton emulation fallback |
| MTP x3 | Works |
| 1,048,576 token context | Works |
| FP8 latent KV cache | Works with e4m3fn storage |
| Prefix caching | Works with the fixes in this fork |
| Vision input | Works |
| Tool calls and the `glm47` parser | Work |

FP8 KV against bfloat16 KV on the same system:

| Test | FP8 KV | bfloat16 KV |
| --- | --- | --- |
| KV pool | 6.67 M tokens | 3.79 M tokens |
| Decode with a 1 M prefix | About 71 tok/s | About 77 tok/s |
| Cold prefill at 1 M | About 305 s | About 173 s |
| Needle recall at 128 K, 512 K, and 1 M | All pass | All pass |

Main changes for this model:

| Area | Change |
| --- | --- |
| Sparse MLA attention | Adds a Triton NoPE kernel for sm_80 and an indexer fallback. |
| FP8 KV stores | Adds software e4m3fn encoding for sm_80. |
| FP8 latent KV cache | Uses uint8 storage and in kernel dequantization without sm_89 FP8 instructions. |
| NVFP4 MoE | Adds the W4A16 path, the Marlin repack holdoff, and a fused Triton emulation fallback. |
| Prefix caching | Adds uncached first allocation, one cached FIFO, transient headroom, and diagnostics. |
| Large KV pools | Fixes integer width in the sparse MLA path. |

## Launch examples

Build one image from this checkout and use it for all three models. Adapt the
GPU list and the layer partition to the target system.

DeepSeek-V4-Flash-Vision-Exp on four cards. This is the tested layout, and the
partition and the two image limit are the values the service validates
against:

```bash
docker run -d --name vllm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_PP_LAYER_PARTITION=12,12,12,7 \
  -e VLLM_MARLIN_FP8_DEQUANT_BF16=1 \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e HF_HUB_OFFLINE=1 \
  -v /path/to/DeepSeek-V4-Flash-Vision-Exp:/model:ro \
  --shm-size=16g -p 8098:8000 \
  vllm-sm80:latest vllm serve /model \
  --served-model-name DeepSeek-V4-Flash-Vision-Exp \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 \
  --block-size 256 --max-model-len 1048576 \
  --max-num-batched-tokens 2048 --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --disable-chunked-mm-input \
  --limit-mm-per-prompt '{"image":2}' \
  --mm-processor-cache-gb 4 \
  --enable-prefix-caching \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --enable-prompt-tokens-details \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}'
```

GLM-5.3-Flash on five cards:

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
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --prefix-match-unit 256
```

Qwen3.8-Flash-Next on four cards:

```bash
docker run -d --name vllm --runtime=nvidia --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  -e VLLM_MARLIN_REPACK_HOLDOFF=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v /path/to/Qwen3.8-Flash-Next-FP8:/model:ro \
  -p 8099:8000 \
  vllm-sm80:latest vllm serve /model \
  --served-model-name Qwen3.8-Flash-Next \
  --pipeline-parallel-size 4 --block-size 256 \
  --max-model-len 1000000 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --enable-chunked-prefill \
  --no-async-scheduling --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
```

Every launch above reports `context_window` in `/v1/models` next to
`max_model_len`. Clients that only read `context_window` otherwise fall back to
262,144 and truncate long prompts.

Settings that are easy to get wrong:

| Setting | Reason |
| --- | --- |
| `--block-size 256` | The DeepSeek and GLM indexer needs a multiple of 128, and the sparse MLA backend needs a multiple of 64. |
| `VLLM_PP_LAYER_PARTITION` | The last rank also carries `lm_head` and the draft layer, so give it fewer decoder layers. DeepSeek Vision is validated at `12,12,12,7` (PP4) and `9,9,9,9,7` (PP5). |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | Align it to the model's hybrid block size. DeepSeek uses 4096. GLM uses 143360 with fp8 KV and 73728 with bfloat16 KV. |
| `VLLM_MARLIN_REPACK_HOLDOFF` | Avoids a load time MMU fault seen on CMP 170HX with driver 610.43.02. Set it to 0 elsewhere. |
| `--kv-cache-memory` | GLM only. Stops Mamba state copies from evicting every hashed checkpoint when the default pool is too small. |
| `--max-num-seqs` | For Qwen this is the real concurrency ceiling. Streams above it only queue. |
| `VLLM_APC_HEADROOM_BLOCKS` | Keeps 32 blocks without hashes for state copies. Set it to 0 on tiny pools, including when running the prefix cache unit tests. |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Illegal memory access during weight load, in the Marlin repack | A caching allocator and driver 610.43.02 interaction on CMP 170HX. Set `VLLM_MARLIN_REPACK_HOLDOFF=1`. It empties the cache on entry, holds every temporary alive for the call, and synchronizes each iteration. The synchronize is load bearing. |
| GPUs stop creating CUDA contexts after a crash | Reload the driver modules (`rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia`, then `modprobe nvidia`) rather than `nvidia-smi --gpu-reset`. On this stack the reset has made things worse. Kill anything holding `/dev/nvidia*` first, `nvtop` included. |
| Engine dead but the HTTP server still answers | The worker died and executor shutdown stalled. `restart: always` cannot fire because the container is still up. Restart it by hand. |
| Prefix cache hits collapse to zero after several long prompts | The pool is too small for the working set, so state copies evict hashed checkpoints. Raise `--kv-cache-memory`. |
| Prefix cache unit tests fail on small pools | Set `VLLM_APC_HEADROOM_BLOCKS=0`. The default of 32 reserved blocks is larger than the pools those tests build. |
| Long prompts crash only once the pool grows | Fixed here. Sparse MLA offsets used to overflow int32 above 4,194,304 cache rows. Upstream kernels carry the same latent bug and only avoid it with smaller pools. |

## Known limits

| Item | Detail |
| --- | --- |
| KDA numerics | Some GLM tests differ from the reference by about 7 percent. |
| Hybrid KV capacity | Mamba state pages alias into larger blocks, so effective capacity is below the raw allocation. |
| FP8 KV prefill | About 1.8 times slower than bfloat16 KV. Decode speed is similar and the larger cache avoids repeated prefill. |
| Qwen host RAM | About 63 GB for the container, most of it the PLE table. Do not cold start two engines at once. |
| Qwen MTP | Not implemented for pipeline parallelism, upstream included. |
| No peer to peer | These cards stage GPU to GPU through host RAM at about 3 GB/s, so never use tensor parallelism. |
| Prefix cache tests | `tests/v1/core/test_prefix_caching.py` has 17 failures on this branch. Six come from the `VLLM_APC_HEADROOM_BLOCKS` default and clear at 0. The rest are unexplained and predate the current work. |
| NIXL connector | `register_kv_caches` has undefined names left from a merge. Nothing here passes `--kv-transfer-config`. |

## Attribution

| Work | Author |
| --- | --- |
| GLM-5.3-Flash support, vLLM PR #53906 | ZJY0516 |
| Qwen3.8-Flash-Next support, vLLM PR #53899 | the PR authors |
| Mamba align boundary fix, vLLM PR #53479 | kamb-code |
| Scheduler speculative decode padding fix, vLLM PR #53962 | njhill |
| Ampere GDN guard, vLLM PR #52743 | the PR author |
| CMP 170HX method reference | allover326/deepseek-v4-cmp170hx |

## History and maintenance

This fork branches from upstream vLLM at `648b7468b`. Everything above that
commit is either this fork's work or a cherry pick named in the attribution
table.

One active branch, `main`. Pre consolidation tips are kept as
`archive/*-20260901` tags.

This is a personal production fork. When upstream vLLM provides equivalent
sm_80 support, this repository will point at the upstream implementation.
