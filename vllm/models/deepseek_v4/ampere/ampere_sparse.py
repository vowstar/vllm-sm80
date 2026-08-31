# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM8x DeepSeek V4 sparse MLA.

Ampere has no FlashMLA and no fp8e4nv, so the FlashMLA/FlashInfer classes
cannot run here. The ROCm AITER path is pure Triton and portable, so CUDA
Ampere reuses it wholesale and only overrides the backend name and the
capability gate.
"""

from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """SM8x DeepSeek V4 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV4AmpereMLASparseBackend
