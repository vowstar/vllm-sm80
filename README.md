# DeepSeek V4 Flash Vision on sm_80

This branch adds an experimental vLLM path for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on NVIDIA sm_80 GPUs.

The base is [`f24af68a2`](https://github.com/vowstar/vllm-sm80/commit/f24af68a270eb72c088004945f3d9642db88b5a9). The runtime candidate is [`bc3da91c8`](https://github.com/vowstar/vllm-sm80/commit/bc3da91c83889ecb228da3dc86a4f145a1abb016).

## Status

The source candidate and immutable image exist. Hardware acceptance is pending. Do not treat this branch as a production support claim.

| Check | State |
| --- | --- |
| Vision source implementation | Complete at `bc3da91c8` |
| Immutable candidate image | Built, not promoted |
| Full checkpoint load | Pending |
| PP4 boot on four CMP 170HX cards | Pending |
| Image understanding and OCR | Pending |
| 1,048,576 token YaRN context | Pending |
| DSpark and MTP | Pending live checks |
| KV offload | Pending live checks |
| Prefix cache hit, eviction, and refill | Pending live checks |
| Sustained hardware workload | Pending on target hardware |

## Implemented changes

| Area | Change |
| --- | --- |
| Model support | Adds the Vision tower, multimodal processor, model registration, and weight loading. |
| Image routing | Routes image tokens through the `bias_vl` MoE path. |
| PrefixLM | Adds full multimodal placeholders and image-local PrefixLM attention. |
| Ampere | Adds the sm_80 sparse sliding attention path for image tokens. |
| Pipeline parallelism | Carries vision metadata across PP ranks. |
| Speculative decoding | Adds the DeepSeek V4 DSpark path for pipeline parallel execution. |
| Embedding safety | Guards image token lookups before text embedding access. |
| Tests | Adds model, multimodal, PrefixLM, router, and sm_80 coverage. |

The implementation contains eight commits and changes 28 files. It adds 2,055 lines and removes 76 lines.

## Target acceptance

The acceptance target uses four Gen2 x16 CMP 170HX 64 GB cards with PP4. A fifth Gen2 x8 card stays idle.

| Area | Required result |
| --- | --- |
| Provenance | The image ID, source revision, source tree, model files, host, boot ID, container, and GPU order must match. |
| Vision | OCR and image understanding must use the real Vision tower. |
| Context | Three needle checks must pass at 1,048,576 tokens with the expected YaRN configuration. |
| Acceleration | DSpark must accept draft tokens and improve decode speed. |
| KV offload | Disabled, native, and restored modes must run in fresh containers with observed store and load counters. |
| Prefix cache | Hit, eviction, and refill checks must pass with independent markers. |
| Stability | A fixed 600 second workload must complete without request failures, Xid, AER, or container restarts. |

After acceptance passes, this README will include measured results and the promoted image ID.

## Repository layout

This repository keeps one branch for each sm_80 model port.

| Branch | Purpose |
| --- | --- |
| `glm53-sm80` | Tested GLM-5.3-Flash production port |
| `dsv4-vision-exp` | DeepSeek V4 Flash Vision candidate and acceptance work |

The repository URL remains [`vowstar/vllm-sm80`](https://github.com/vowstar/vllm-sm80). Runtime deployment files live outside this source repository.
