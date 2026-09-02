# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen Triton kernels from the loaded model's compile keys."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen4_exp",
        "qwen4_exp_text",
    }
)

# Covers L=1 constexpr, non-divisible runtime L, and divisible runtime L.
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)


@dataclass(frozen=True)
class _QwenGDNWarmupConfig:
    h: int
    hv: int
    k: int
    v: int
    conv_kernel_size: int
    conv_state: torch.Tensor
    conv_dtype: torch.dtype
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_stride_token: int
    state_dtype: torch.dtype

    @property
    def conv_dim(self) -> int:
        return 2 * self.h * self.k + self.hv * self.v


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _is_qwen_gdn_layer(module: object) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel_size",
            "tp_size",
            "kv_cache",
            "A_log",
            "dt_bias",
        )
    )


def _iter_qwen_gdn_layers(static_forward_context: object):
    if not isinstance(static_forward_context, dict):
        return

    for module in static_forward_context.values():
        if _is_qwen_gdn_layer(module):
            yield module


def _split_qwen_gdn_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        conv_cache, ssm_state = kv_cache[:2]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state

    if isinstance(kv_cache, torch.Tensor) and kv_cache.size(0) >= 2:
        conv_cache = kv_cache[0]
        ssm_state = kv_cache[1]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state
    return None


def _qwen_gdn_warmup_config(
    static_forward_context: object,
) -> _QwenGDNWarmupConfig | None:
    found_layer = False
    for layer in _iter_qwen_gdn_layers(static_forward_context):
        found_layer = True
        cache_tensors = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache_tensors is None:
            continue

        conv_cache, ssm_state = cache_tensors
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        tp_size = int(layer.tp_size)
        h = int(layer.num_k_heads) // tp_size
        hv = int(layer.num_v_heads) // tp_size

        return _QwenGDNWarmupConfig(
            h=h,
            hv=hv,
            k=int(layer.head_k_dim),
            v=int(layer.head_v_dim),
            conv_kernel_size=int(layer.conv_kernel_size),
            conv_state=conv_state,
            conv_dtype=conv_state.dtype,
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_stride_token=int(ssm_state.stride(0)),
            state_dtype=ssm_state.dtype,
        )

    if found_layer:
        logger.info("Skipping Qwen GDN Triton warmup: no bound Qwen GDN cache found.")
    else:
        logger.info("Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.")
    return None


def _warm_causal_conv1d_fwd_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    x_storage = torch.empty(
        (1, config.conv_dim), dtype=config.conv_dtype, device=device
    )
    x = x_storage.t()
    weight = torch.empty(
        (config.conv_dim, config.conv_kernel_size),
        dtype=config.conv_dtype,
        device=device,
    )
    cache_indices = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    causal_conv1d_fn(
        x,
        weight,
        None,
        config.conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        metadata=None,
        validate_data=False,
    )


def _warm_fused_post_conv_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E501
        fused_post_conv_prep,
    )

    qkv_dim = 2 * config.h * config.k + config.hv * config.v
    for length in _FLA_POST_CONV_WARMUP_LENGTHS:
        conv_output = torch.empty(
            (length, qkv_dim), dtype=config.conv_dtype, device=device
        )
        a = torch.empty((length, config.hv), dtype=config.conv_dtype, device=device)
        b = torch.empty_like(a)

        fused_post_conv_prep(
            conv_output,
            a,
            b,
            config.a_log,
            config.dt_bias,
            config.h,
            config.k,
            config.v,
            apply_l2norm=True,
            output_g_exp=False,
        )


def _warm_fused_sigmoid_gating_delta_rule_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device)
    a = torch.empty((1, 1, config.hv), dtype=config.conv_dtype, device=device)
    b = torch.empty_like(a)
    state = torch.empty(
        (1, config.state_stride_token),
        dtype=config.state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
    ssm_state_indices.zero_()

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=a,
        b=b,
        dt_bias=config.dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


def _find_qsa_indexer(runner: "GPUModelRunner"):
    model = getattr(runner, "model", None)
    if model is None:
        return None
    for module in model.modules():
        if all(
            hasattr(module, attr)
            for attr in (
                "index_n_heads",
                "index_head_dim",
                "compress_ratio",
                "token_topk",
                "raw_key_cache",
                "compressed_key_cache",
                "rotary_emb",
                "q_layernorm",
                "k_layernorm",
            )
        ):
            return module
    return None


def _warm_qsa_kernels(
    runner: "GPUModelRunner",
    model_config: object,
    device: torch.device,
    static_forward_context: object,
) -> None:
    """Pre-compile the four QSA Triton kernels on the shapes the scheduler
    produces, so none of them JIT inside a pipeline-parallel collective.

    The warmup ladder (vllm/v1/worker/gpu/warmup.py) runs prefill with a
    prompt a few tokens long, so it never reaches the large-shape
    specializations these kernels pick at real batch and chunk sizes. This
    walks the shapes directly instead, reusing the layer's already-bound
    cache tensors for exact page sizes and strides.
    """
    indexer = _find_qsa_indexer(runner)
    if indexer is None:
        logger.info("Skipping QSA Triton warmup: no QSA indexer layer found.")
        return

    n_heads = int(indexer.index_n_heads)
    head_dim = int(indexer.index_head_dim)
    compress_ratio = int(indexer.compress_ratio)
    token_topk = int(indexer.token_topk)
    output_width = token_topk + compress_ratio - 1

    raw_cache = indexer.raw_key_cache.kv_cache
    comp_cache = indexer.compressed_key_cache.kv_cache
    max_len = int(getattr(model_config, "max_model_len", 0) or 0)
    # The block table row length is cdiv(max_len, cache block_size) -- the
    # scheduler's storage block size, NOT the compressed cache's state count
    # (comp_cache.shape[1] is the per-block state count, 14336 here, which
    # would shrink the table to 70 rows and compile the wrong specialization).
    cache_block_size = int(getattr(runner.cache_config, "block_size", 0) or 0)
    page_table_width = (
        (max_len + cache_block_size - 1) // cache_block_size
        if max_len and cache_block_size else 0
    )

    hf_text_config = getattr(model_config, "hf_text_config", None)
    num_q_heads = int(getattr(hf_text_config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(hf_text_config, "num_key_value_heads", 0) or 0)
    main_head_dim = int(getattr(hf_text_config, "head_dim", 0) or 0)
    logger.info(
        "QSA warmup config: indexer heads=%d head_dim=%d ratio=%d topk=%d "
        "page_table_width=%d cache_block_size=%d main(q=%d,kv=%d,hd=%d).",
        n_heads, head_dim, compress_ratio, token_topk, page_table_width,
        cache_block_size, num_q_heads, num_kv_heads, main_head_dim,
    )
    num_reqs = 1
    bf16 = torch.bfloat16

    # Kernels 2+3: _qsa_mqa_paged_kernel and _expand_qsa_indices_kernel.
    # rows <= 32 and rows > 32 select different TILES_PER_PROG specializations.
    # columns = page_table_width * page_size (~8M here) would push the launch
    # grid past CUDA's 65535 Y limit at rows <= 32. columns is a runtime arg,
    # not a constexpr, so clamping it via num_columns keeps the grid legal
    # without changing the compiled specialization.
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        expand_qsa_block_indices_cuda,
        qsa_mqa_paged,
        qsa_sparse_paged_attention,
    )

    block_topk = token_topk // compress_ratio
    for rows in (1, 33):
        q = torch.empty((rows, n_heads, head_dim), dtype=bf16, device=device)
        page_table = torch.zeros(
            (num_reqs, page_table_width), dtype=torch.int32, device=device
        )
        token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
        # query_positions is logical_positions on the real path, int64.
        query_positions = torch.zeros(rows, dtype=torch.int64, device=device)
        sequence_lengths = torch.ones(num_reqs, dtype=torch.int32, device=device)
        qsa_mqa_paged(
            q,
            comp_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
            num_columns=4096,
        )
        block_indices = torch.zeros(
            (rows, block_topk), dtype=torch.int32, device=device
        )
        out = torch.zeros((rows, output_width), dtype=torch.int32, device=device)
        expand_qsa_block_indices_cuda(
            block_indices,
            query_positions,
            sequence_lengths,
            token_to_req,
            compress_ratio,
            token_topk,
            out,
        )

    # Kernel 4: _qsa_sparse_paged_gqa_splitk_kernel. Its NUM_SPLITS / BLOCK_N
    # are picked from base_programs = rows * num_kv_heads, so walk every
    # bucket boundary.
    if num_q_heads and num_kv_heads and main_head_dim and page_table_width:
        k_cache = torch.empty(
            (16, cache_block_size, num_kv_heads, main_head_dim), dtype=bf16, device=device
        )
        v_cache = torch.empty_like(k_cache)
        for rows in (1, 2, 8, 16, 128, 256, 300):
            q = torch.empty(
                (rows, num_q_heads, main_head_dim), dtype=bf16, device=device
            )
            logical_indices = torch.zeros(
                (rows, output_width), dtype=torch.int32, device=device
            )
            block_table = torch.zeros(
                (num_reqs, page_table_width), dtype=torch.int32, device=device
            )
            token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
            qsa_sparse_paged_attention(
                q, k_cache, v_cache, logical_indices, block_table, token_to_req
            )

    # Kernel 1: _qsa_pre_indexer_kernel. TILE_T_Q/TILE_H_Q split on
    # num_tokens <= 4096, so warm both sides. num_k_work = 0 compiles the whole
    # kernel (both the K-compress and Q-normalize branches) while executing only
    # the Q branch, so no compressor metadata needs to be built here.
    from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import qsa_pre_indexer

    mrope_section = getattr(indexer.rotary_emb, "mrope_section", None)
    rope_pos_offset = (
        indexer.raw_key_cache.rope_position_offset
        if indexer.raw_key_cache.rope_position_cache is not None
        else None
    )
    cos_sin_cache = indexer.rotary_emb.cos_sin_cache
    q_norm_weight = indexer.q_layernorm.weight
    k_norm_weight = indexer.k_layernorm.weight
    eps = float(indexer.q_layernorm.variance_epsilon)
    for num_tokens in (1, 5000):
        q = torch.empty(
            (num_tokens, n_heads * head_dim), dtype=bf16, device=device
        )
        k = torch.empty((num_tokens, head_dim), dtype=bf16, device=device)
        if mrope_section is not None:
            positions = torch.zeros(
                (3, num_tokens), dtype=torch.int64, device=device
            )
        else:
            positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        q_out = torch.empty(
            (num_tokens, n_heads, head_dim), dtype=bf16, device=device
        )
        state_slots = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        state_block_table = torch.zeros(
            (num_reqs, 1), dtype=torch.int32, device=device
        )
        query_start_loc = torch.zeros(
            num_reqs + 1, dtype=torch.int32, device=device
        )
        logical_positions = torch.zeros(
            num_tokens, dtype=torch.int64, device=device
        )
        compressed_slots = torch.zeros(
            num_tokens, dtype=torch.int64, device=device
        )
        k_work_metadata = torch.empty((0, 2), dtype=torch.int32, device=device)
        qsa_pre_indexer(
            q,
            k,
            positions,
            cos_sin_cache,
            q_norm_weight,
            k_norm_weight,
            eps,
            q_out,
            raw_cache,
            state_slots,
            state_block_table,
            query_start_loc,
            logical_positions,
            comp_cache,
            compressed_slots,
            k_work_metadata,
            compress_ratio=compress_ratio,
            mrope_section=mrope_section,
            rope_pos_offset=rope_pos_offset,
        )
    torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_triton_warmup(
    runner: "GPUModelRunner",
    model_config: object,
) -> None:
    """Warm Qwen Triton kernels reported by the JIT monitor."""
    if runner.is_pooling_model:
        return

    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_type = None
    for config in (hf_text_config, hf_config):
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            model_type = str(model_type)
            break
    if model_type not in _QWEN_MODEL_TYPES:
        return

    device = getattr(runner, "device", torch.device("cuda"))
    logger.info("Warming up Qwen Triton kernels for model_type=%s.", model_type)

    compilation_config = getattr(runner, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    gdn_config = _qwen_gdn_warmup_config(static_forward_context)
    try:
        if gdn_config is not None:
            _warm_causal_conv1d_fwd_kernel(device, gdn_config)
            _warm_fused_post_conv_kernel(device, gdn_config)
            _warm_fused_sigmoid_gating_delta_rule_update_kernel(device, gdn_config)
        _warm_qsa_kernels(runner, model_config, device, static_forward_context)
    except Exception:
        logger.warning(
            "Qwen Triton warmup did not complete; some kernels may JIT on the "
            "first request.",
            exc_info=True,
        )
    _synchronize_device(device)
