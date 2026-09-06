# vllm-sm80

This vLLM fork runs three large models on NVIDIA sm_80 GPUs that upstream vLLM
does not support there. All three share one `main` branch and one image.

Development and testing use CMP 170HX 64 GB cards with pipeline parallelism and
driver 610.43.02. Other sm_80 GPUs such as A100 are not tested.

## Status

| Model | State | Tested layout |
| --- | --- | --- |
| DeepSeek-V4-Flash-Vision-Exp | Serving, measured | 4 cards, PP4 (12,12,12,7), 1M context, vision, DSpark x3, fp8_ds_mla KV, split-K decode LUT |
| Qwen3.8-Flash-Next-FP8 | Serving, measured | 4 cards, PP4, 1M YaRN context, PLE CPU offload |
| GLM-5.3-Flash | Serving, measured | 5 cards, PP5, 1M context, vision, MTP x3, fp8 KV |

DeepSeek-V4-Flash-0731, the text only checkpoint, also loads and answers on
this branch. It was validated at 64K context without speculative decoding.

Since 2026-09-05 one image, `vllm-sm80:unified-84e5971d2a` built from `main`
at `84e5971d2a`, backs every model on both GPU hosts.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | sm_80. Tested only on CMP 170HX 64 GB, which has no peer to peer and a 64 MB BAR1. |
| Cards | 4 for Qwen, 5 for GLM, 4 or 5 for DeepSeek. Tensor parallelism does not work on these cards, so the count is a pipeline depth. |
| VRAM | 64 GB per card. The tightest rank runs within about 1.6 GiB of the limit. |
| Host RAM | DeepSeek costs about 11 GB once serving, but weight load peaks far higher, so leave 30 GB free. Qwen needs about 63 GB, because its PLE table is 51 GB and lives in host memory. |
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

All three columns come from one harness against a live service with real
technical prose as the prompt.

**Benchmark convention.** From 2026-09-05, DeepSeek and Qwen benchmarks are
pipeline parallel 4 with layer partition `12,12,12,7` on four CMP 170HX 64 GB
cards at PCIe Gen2 x16, so numbers stay comparable across hosts and over
time. GLM is benchmarked at its minimum viable topology PP5, because PP4
does not fit on a 64 GiB CMP 170HX (see Known limits). Each DeepSeek table
names the topology it was measured on, and older PP5 numbers are marked
historical rather than mixed into PP4 tables. The harness is
`performance_matrix_real.py`. Single-point decode numbers carry about ±20
percent run-to-run noise from DSpark draft acceptance variance, so only a
same-day same-harness A/B is exactly comparable. Concurrency tables quote
aggregate throughput, which charges time to first token and pipeline fill
against the result. The steady generation rate, defined further down,
excludes both.

Use real prose. A prompt built from one repeated word makes speculative
decoding accept almost everything at short context and almost nothing at long
context, which turns a flat curve into a cliff. The same DeepSeek service
measured 101 tok/s at 2 K and 29 tok/s at 1 M on a repeated word prompt, and
76 tok/s and 26 tok/s on prose. Only the prose numbers mean anything.

### Decode speed against context length

One stream, cold prompt, no prefix cache hit. Time to first token is the full
prefill. Topologies in this table: DeepSeek PP4 on the EPYC 7282 host, Qwen
PP4 on the Ryzen 9 5900X host, GLM PP5.

| Prompt tokens | DeepSeek tok/s | Qwen tok/s | GLM tok/s |
| ---: | ---: | ---: | ---: |
| About 2 K | 85.2 | 53.3 | 68.7 |
| About 7 K | 71.7 | 51.4 | 60.8 |
| About 30 K | 73.9 | 52.3 | 64.3 |
| About 90 K | 70.4 | 52.0 | 62.7 |
| About 230 K | 83.8 | 53.0 | 62.5 |
| About 460 K | 73.6 | 55.1 | 54.0 |
| About 900 K to 1 M | 67.7 | 54.1 | 68.3 |

Read the three columns as three attention designs, not as a ranking.

Qwen holds one decode rate at every length because its GDN linear attention
keeps a constant size recurrent state. GLM is also close to flat because only
11 of its 45 layers are sparse MLA and the other 34 are KDA linear attention.
DeepSeek used to be the one that fell away at long context; after the split-K
sparse decode merge (ported from
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport), see the
DeepSeek section below) its curve is nearly flat too, 68 to 85 tok/s from 2 K
to 850 K, and it leads the other two at almost every length measured.

Two caveats on this table. Only DeepSeek and GLM run speculative decoding, so
part of their short context advantage is draft acceptance rather than step
time. The GLM column uses fp8 KV, the production default, so it decodes a few
tokens per second slower than the same model with bfloat16 KV; the comparison
table below quantifies the gap.

### Unified image on both hosts (2026-09-05)

DeepSeek-V4-Flash-Vision-Exp on `vllm-sm80:unified-84e5971d2a`, PP4
12,12,12,7, DSpark x3, fp8_ds_mla KV with the split-K decode LUT. Host A has
an EPYC 7282, host B has a Ryzen 9 5900X, and both run four CMP 170HX 64 GB
cards at Gen2 x16. Same harness, same day, one stream, cold prompt.

| Prompt tokens | Host A tok/s | Host B tok/s |
| ---: | ---: | ---: |
| About 2 K | 61.3 | |
| About 30 K | 94.0 | 100.8 |
| About 120 K | 72.0 | |
| About 480 K | 88.5 | 61.6 |

| Streams | Host A aggregate | Host B aggregate |
| ---: | ---: | ---: |
| 1 | 71.6 tok/s | |
| 16 | 364.7 tok/s | 310.8 tok/s |

Host B also ran its previous image the same day, which isolates the image
change on one machine:

| Measurement | Old image | Unified image |
| --- | ---: | ---: |
| Decode at about 30 K | 75.8 tok/s | 100.8 tok/s |
| Decode at about 120 K | 53.1 tok/s | |
| Decode at about 480 K | 45.9 tok/s | 61.6 tok/s |
| 16 streams aggregate | 249.7 tok/s | 310.8 tok/s |

Host A ran the same A/B in the same campaign. Decode at about 120 K went
from 51.7 to 84.1 tok/s and at about 480 K from 43.4 to 88.5 tok/s, roughly
double at long context. Each single point still carries the ±20 percent
acceptance noise from the convention note above.

### Prefill

Time to first token on the same runs, with the rate it implies.

| Prompt tokens | DeepSeek | Qwen | GLM |
| ---: | ---: | ---: | ---: |
| About 2 K | 0.9 s, 2,166 tok/s | 0.5 s, 4,102 tok/s | 1.2 s, 1,439 tok/s |
| About 30 K | 5.7 s, 5,333 tok/s | 2.9 s, 10,267 tok/s | 8.8 s, 3,089 tok/s |
| About 90 K | 23.3 s, 5,177 tok/s | 6.2 s, 14,499 tok/s | 20.9 s, 3,973 tok/s |
| About 230 K | 55.4 s, 4,352 tok/s | 15.3 s, 14,670 tok/s | 49.1 s, 4,216 tok/s |
| About 460 K | 167.6 s, 2,878 tok/s | 33.9 s, 13,277 tok/s | 102.1 s, 4,055 tok/s |
| About 900 K to 1 M | 349.9 s, 2,235 tok/s | 82.9 s, 10,751 tok/s | 289.0 s, 3,177 tok/s |

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
| Cards, pipeline stages | 4, PP4 | 5, PP5 | 5, PP5 |
| Weights per card | 41.2 GiB | About 25 GiB | About 49 to 64 GiB |
| KV pool | 2,674,615 tokens | 7,244,396 tokens | 6,670,108 tokens |
| Concurrency at 1 M context | 2.55x | 6.91x | 6.36x |
| Cold start to serving | 4 minutes from NVMe | 35 minutes from spinning disks | 4 minutes from NVMe |

The Qwen cold start is dominated by reading 173 GiB of weights, 51 GB of it
the PLE table, off spinning disks. The same model on NVMe would not take that long.
GLM holds a much larger KV pool because its 34 KDA layers store a fixed size
recurrent state instead of a growing cache, and because its launcher pins
`--kv-cache-memory` at 12.5 GiB per rank. Qwen reaches its pool with
`--kv-cache-memory` pinned at 29 GiB per rank plus a 150 GiB CPU offload tier
(the native connector, ported from upstream #54743 and #55033); its util-derived
pool is only 5.66 M tokens.

### Multi-prefix cache residency (Qwen)

Independent 200 K-token prefixes, filled then replayed, PP5, bfloat16 KV. A
prefix counts as resident when the replay reports at least 95 percent cached
prompt tokens.

| Configuration | Resident prefixes | What happens past the limit |
| --- | ---: | --- |
| util-derived pool (5.66 M tokens) | 3 | the 4th replay collapses to 0 cached tokens |
| pin 29 GiB/rank + 150 GiB CPU offload (7.24 M tokens) | 4 | the 5th degrades gracefully to 86 percent, no cascade |

Single-prefix replay quality improved at the same time: a 200 K prefix went
from 86.01 percent cached (172,032 tokens, 4.1 s probe) to 99.97 percent
(199,936 tokens, 1.35 s probe), and mean TTFT in a 3x200K mix dropped from
6.99 s to 3.72 s. Decode speed is unchanged by the pin and the offload tier
(99.68 vs 100.45 tok/s aggregate at 3x200K, inside run-to-run noise). An fp8
QSA KV cache was also measured: it grows the pool 1.82x but makes cold
prefill about 9x slower and buys only about 33 percent more residency, so
production stays on bfloat16 KV.

### Throughput against concurrency

The three models run different topologies (in this table DeepSeek is PP4,
Qwen PP4, GLM PP5) and different true-concurrency caps (`--max-num-seqs`), so
one "streams" count does not mean the same work per model. One harness, 512
output tokens, about 1,600-token prose prompts with unique prefixes, against
each live service:

| Concurrency | Qwen aggregate | Qwen steady | GLM aggregate | DeepSeek aggregate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 39.7 tok/s | 39.6 tok/s | 63.8 tok/s | 76.1 tok/s |
| 4 | 131.3 tok/s | 142.4 tok/s | 153.0 tok/s | 180.8 tok/s |
| 8 | 221.9 tok/s | 252.8 tok/s | 188.3 tok/s | 253.9 tok/s |
| 16 | 375.5 tok/s | 432.5 tok/s | 309.1 tok/s | 344.7 tok/s |
| 32 | 506.0 tok/s | 644.3 tok/s | 432.8 tok/s | 420.3 tok/s |

Per-request median at the same points: Qwen 39.7 / 32.8 / 27.8 / 23.5 / 15.8,
GLM 63.8 / 39.0 / 23.9 / 20.0 / 14.0, DeepSeek 78.1 / 61.6 / 40.3 / 28.3 /
14.9 tok/s. Qwen is slowest single-stream because it is the only one without
speculative decoding, but scales best (12.7x from 1 to 32 streams) because its
GDN linear attention batches cheaply; GLM and DeepSeek scale about 6.5x.

A newer Qwen PP4 run on the EPYC 7282 host (2026-09-06, after further warmup
fixes) reaches 628.4 tok/s aggregate at 32 streams, 24 percent above the
506.0 in the table above. The full ladder is in the Qwen section.

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
requests are in flight. The table above uses 512-token outputs, which is too
short to fill that pipeline, so its aggregate column understates the steady
rate everywhere. Run the same 8-stream point for longer and the two converge:

| Concurrency | Output/req | Wall | aggregate | steady generation | per-request median |
| --- | --- | --- | --- | --- | --- |
| 8 | 512 | 18.5 s | 221.9 tok/s | 252.8 tok/s | 27.8 tok/s |
| 8 | 2048 | 64.6 s | 253.7 tok/s | 259.1 tok/s | 31.7 tok/s |

At 2048 tokens per request the gap is 2 percent. Any comparison that puts one
system's steady dashboard reading next to another system's short-run aggregate
is measuring output length, not the engine.

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

A separate fix was also needed at high concurrency. The PLE offload producer
used a one-entry queue, sized for `pipeline_parallel_size=1` where each forward
consumes its output before the next launch. Pipeline parallelism keeps several
forwards in flight, so the queue filled and rank 0, the only rank holding the
PLE embedding, took the service down. Blocking alone only defers that to the
60 s timeout, so the queue is now sized `2 * pipeline_parallel_size`.

## DeepSeek V4 Flash and Vision

The Vision checkpoint is `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.

| Feature | State |
| --- | --- |
| 1,048,576 token context | Works |
| Image input | Works, checked against a generated test image |
| DSpark speculative decoding under PP | Works |
| Prefix caching | Works, 99.7 percent replay after the unified cutover |
| Tool calls and the `deepseek_v4` parser | Works, checked with a function call |
| KV offload | Present, not exercised here |

Production on both hosts runs `vllm-sm80:unified-84e5971d2a`, built from
`main` at `84e5971d2a`, at PP4 with partition `12,12,12,7`, DSpark x3, and
fp8_ds_mla KV with the split-K decode LUT. The KV pool is 2,674,615 tokens on
the EPYC host and 2,396,420 tokens on the Ryzen host, that is 2.55x and 2.29x
concurrent 1M-token requests. After the cutover on the Ryzen host, factual QA
passed and a prefix replay cached 102,144 of 102,441 prompt tokens (99.7
percent) with the replay wall dropping from 21.2 s to 0.5 s. The two hosts
use different docker graph drivers, so cross-host image acceptance compares
the 30 RootFS diff_ids, not the image ID.

The text-only 0731 checkpoint on the same PP4 layout with the LUT build:
82.5 tok/s decode at about 120 K prompt tokens, 67.8 tok/s at about 480 K,
377.8 tok/s aggregate at 32 streams, mean DSpark acceptance length 3.35.

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
| Sparse decode | Ports the split-K sparse decode kernels, fp8 LUT dequantization, deterministic CUDA top-k, prefill tiling, and the fp8_ds_mla planar layout fix from [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) (branch `wtd-merge-20260905`). The long context decode curve is nearly flat after this, 68 to 85 tok/s from 2 K to 850 K. Their blocked prefill kernel is present but gated off by default; its first live run hung the engine and the current hypothesis is first-use Triton JIT stalling a pipeline collective. |

## Qwen3.8-Flash-Next

The checkpoint is `Qwen3.8-Flash-Next-FP8` with the `qwen4_exp` architecture.
Upstream [PR #53899](https://github.com/vllm-project/vllm/pull/53899) validated
tensor parallel and data parallel only, so the pipeline parallel path here is
this fork's work.

| Feature | State |
| --- | --- |
| PP4 serving | Works |
| 1,048,576 token YaRN context | Works, from a native 262,144 |
| PLE CPU offload under PP | Works |
| 8 concurrent streams | Works, 253 tok/s steady generation, cold prefix cache |
| 16 concurrent streams | Works, 432 tok/s steady generation |
| 32 concurrent streams | Works, 644 tok/s steady generation |
| Prefix caching | Works |
| Vision tower | Loads and warms up, image accuracy not checked |
| MTP | Not available under PP, upstream included |

The PLE table is a 51 GB FP8 ngram embedding of 16 heads over a 20 million
entry vocabulary, which is 51 of the model's 180 billion parameters. It stays in host memory and costs about microseconds per
token, so it is not the decode bottleneck. It does need roughly 63 GB of host
RAM for the whole container.

PP4 on the EPYC 7282 host with the unified image, measured 2026-09-06: four
CMP 170HX at Gen2 x16, a 20 GiB KV pin per rank, KV pool 3,337,985 tokens, a
150 GiB CPU KV offload tier, and the full 1,048,576 token YaRN window. No
speculative decoding. One stream, cold prompt:

| Prompt tokens | Decode tok/s | TTFT |
| ---: | ---: | ---: |
| About 2 K | 36.1 | 0.9 s |
| About 7 K | 34.8 | 1.4 s |
| About 30 K | 36.2 | 2.7 s |
| About 118 K | 36.9 | 7.5 s |
| About 472 K | 38.1 | 29.4 s |

| Streams | Aggregate tok/s | Per-request median tok/s |
| ---: | ---: | ---: |
| 1 | 36.2 | 36.9 |
| 4 | 82.2 | 31.3 |
| 8 | 234.8 | 30.2 |
| 16 | 391.0 | 25.3 |
| 32 | 628.4 | 20.5 |

The decode curve is flat because GDN linear attention batches cheaply. The
32-stream aggregate of 628.4 tok/s beats the 2026-09-02 same-host PP4 figure
of 506.0 by 24 percent, thanks to the warmup and `do_not_specialize` fixes
in between. Single-request decode reads about 36 tok/s on this EPYC 7282
(Zen2) host against about 53 tok/s on the Ryzen 9 5900X host, because the
PLE ngram lookup runs on the host CPU.

Main changes for this model:

| Area | Change |
| --- | --- |
| PLE offload | Enables it on rank 0 under pipeline parallelism instead of rejecting PP. |
| PLE offload | Waits for the staging queue instead of raising `queue.Full` and killing the engine. |
| Model state | Disables ngram state on non first ranks rather than failing. |
| Weight loading | Skips the final mixer weights on non last ranks. |
| Marlin FP8 | Adds the repack holdoff that CMP 170HX needs. |
| QSA kernels | Pre-compiles them in warmup and stops Triton specializing on runtime integers, so none JIT inside a pipeline collective. |

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

Unified image validation, 2026-09-05: the DeepSeek merge changed zero
GLM-critical files, and production on the unified image confirms it. The
smoke suite passes in full, vision and tool_choice included. MTP acceptance
is 0.615. A 195 K prefix replays with 188,160 cached tokens on the first
replay, a 45.5 s fill against a 5.7 s replay wall. At a 5.2 K prompt, TTFT
is 3.49 s and decode runs 45.8 to 52.2 tok/s.

PP5 on the EPYC 7282 (Zen2) host with the unified image, measured 2026-09-06
against the live production service: PP5 11,9,9,9,7 on five CMP 170HX at
Gen2 x16, fp8 KV, MTP x3, max-model-len 1,048,576, KV pool 6,670,108
tokens. One stream, cold prompt:

| Prompt tokens | Decode tok/s | TTFT |
| ---: | ---: | ---: |
| 1,845 | 52.3 | 1.39 s |
| 7,381 | 46.3 | 4.68 s |
| 29,519 | 46.5 | 8.98 s |
| 117,991 | 46.6 | 27.9 s |
| 471,870 | 44.8 | 115.9 s |
| 943,742 | 49.2 | 263.3 s |

| Streams | Aggregate tok/s | Per-request median tok/s |
| ---: | ---: | ---: |
| 1 | 56.5 | 61.1 |
| 4 | 117.7 | 36.8 |
| 8 | 210.4 | 32.6 |
| 16 | 257.1 | 18.9 |
| 32 | 341.1 | 13.0 |

The decode curve is flat at about 45 to 52 tok/s from 2 K to 1 M. Older GLM
figures in this document (decode about 66 to 74 tok/s, 32-stream aggregate
about 433 tok/s) came from different host hardware and the pre-unification
image, so they are historical and not comparable point for point. A same-day
same-host spot check, old image 54.0 against unified 45.8 to 52.2 tok/s at a
5.2 K prompt, reads as parity inside measurement noise, so the delta against
the older tables is attributed to the host change, not to the unified
image.

GLM PP4 is not viable on a 64 GiB CMP 170HX, measured 2026-09-06 with three
boot attempts. Util-derived KV sizing OOMs deterministically on rank 2:
after the NVFP4 Marlin weights plus the MTP draft, 2.31 GiB is free against
a 3.38 GiB profiling workspace ask. A 1.5 GiB KV pin per rank OOMs at the
identical point. Dropping max-model-len to 262,144 was never reached,
because the repeated OOM crashes escalated into a driver-level fault (NVRM
Xid 154 on sibling cards, recovery action "OS Reboot", the known CMP 170HX
plus driver 610.43.02 cascade family) and the launcher's own safeguard
refused to continue. Recovery needed an nvidia module reload, and
production (GLM PP5 and DeepSeek PP4) was fully restored afterwards. PP5
11,9,9,9,7 is the minimum viable and production topology, so the GLM
benchmark tables stay PP5-labeled as the documented exception to the PP4
benchmark convention.

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

DeepSeek-V4-Flash-Vision-Exp on four cards. This is the tested layout and the
partition is the value the service validates against:

```bash
docker run -d --name vllm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_PP_LAYER_PARTITION=12,12,12,7 \
  -e VLLM_MARLIN_FP8_DEQUANT_BF16=1 \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_DSV4_DECODE_FP8_LUT=1 \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e HF_HUB_OFFLINE=1 \
  -v /path/to/DeepSeek-V4-Flash-Vision-Exp:/model:ro \
  --shm-size=16g -p 8098:8000 \
  vllm-sm80:latest vllm serve /model \
  --served-model-name DeepSeek-V4-Flash-Vision-Exp \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8_ds_mla \
  --block-size 256 --max-model-len 1048576 \
  --max-num-batched-tokens 2048 --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --disable-chunked-mm-input \
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
  --max-model-len 1000000 --max-num-seqs 32 \
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
| `--kv-cache-memory` for Qwen at PP4 | Keep it at or under about 20 GiB per rank. A 31 GiB pin crash-loops the engine at engine-init OOM. |
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
| No peer to peer | These cards stage GPU to GPU through host RAM at about 3 GB/s, so never use tensor parallelism. |
| Prefix cache tests | `tests/v1/core/test_prefix_caching.py` has 17 failures on this branch. Six come from the `VLLM_APC_HEADROOM_BLOCKS` default and clear at 0. The rest are unexplained and predate the current work. |
| NIXL connector | `register_kv_caches` has undefined names left from a merge. Nothing here passes `--kv-transfer-config`. |
| Blocked c128a prefill kernel | Ported with the split-K stack but default OFF (`VLLM_SPARSE_DENSE_QUERY_BLOCK=0`) pending a warmup fix. Its first live run hung the engine, the current hypothesis being first-use Triton JIT inside a pipeline collective. |
| GLM PP4 | Tested 2026-09-06 and impossible on a 64 GiB CMP 170HX with MTP enabled. Util-derived KV sizing and a 1.5 GiB pin per rank both OOM on rank 2 during profiling (2.31 GiB free against a 3.38 GiB workspace ask), and repeated attempts escalated to a driver fault that needed a module reload. PP5 11,9,9,9,7 is the minimum viable and production topology. |
| Qwen KV pin at PP4 | Must not exceed about 20 GiB per rank. A 31 GiB pin crash-loops the engine at engine-init OOM. |

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

2026-09-05: `wtd-merge-20260905` merged into `main` at `84e5971d2a`. One
image now backs every model on both GPU hosts, and the PP4 DeepSeek
benchmark convention took effect.

This is a personal production fork. When upstream vLLM provides equivalent
sm_80 support, this repository will point at the upstream implementation.
