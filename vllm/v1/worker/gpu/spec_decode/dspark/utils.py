# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


def _has_real_weight(module: nn.Module | None) -> bool:
    return (
        module is not None
        and not isinstance(module, PPMissingLayer)
        and getattr(module, "weight", None) is not None
    )


def _validate_last_pp_stage_aux_layers(target_inner: nn.Module, use_pp: bool) -> None:
    if not use_pp:
        return
    aux_layers = tuple(getattr(target_inner, "aux_hidden_state_layers", ()))
    start_layer = int(getattr(target_inner, "start_layer", -1))
    end_layer = int(getattr(target_inner, "end_layer", -1))
    if not aux_layers or any(
        layer_idx <= start_layer or layer_idx > end_layer
        for layer_idx in aux_layers
    ):
        raise RuntimeError(
            "DeepSeek V4 DSpark with pipeline parallelism requires every "
            "auxiliary output on the last PP stage; got "
            f"aux_layers={aux_layers}, stage=({start_layer}, {end_layer}]"
        )


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4 draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type == "deepseek_v4":
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
    )

    use_pp = get_pp_group().world_size != 1
    if use_pp and (
        draft_model_config.hf_config.model_type != "deepseek_v4"
        or not current_platform.is_cuda()
    ):
        raise NotImplementedError(
            "Pipeline-parallel DSpark is currently implemented only for "
            "DeepSeek V4 on CUDA."
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = getattr(target_language_model, "model", target_language_model)
    _validate_last_pp_stage_aux_layers(target_inner, use_pp)

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if (
        not use_pp
        and target_embed is not None
        and draft_model_config.get_vocab_size() <= target_vocab_size
        and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        )
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    if use_pp and not _has_real_weight(target_lm_head):
        raise RuntimeError(
            "DeepSeek V4 DSpark requires a materialized target lm_head on "
            "the last pipeline rank"
        )
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
