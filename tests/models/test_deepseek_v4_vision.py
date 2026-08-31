# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.models.deepseek_v4.multimodal import _position_image_blocks
from vllm.models.deepseek_v4.visibility import (
    get_image_swa_bounds,
    get_max_image_swa_width,
)
from vllm.models.deepseek_v4.vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4VisionAligner,
    DeepseekV4VisionTransformer,
    apply_image_type_embeddings,
    build_image_block,
    grid_tokens,
    preprocess_image,
)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


def _tiny_vision_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=12,
        vision_n_layers=2,
        vision_dim=16,
        vision_n_heads=4,
        vision_inter_dim=24,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        vision_max_n_token=32,
        vision_min_pixels=0,
        vision_max_wh_ratio=8,
    )


def test_image_block_matches_official_n_layout() -> None:
    types, permutation = build_image_block(3, 2, start_pos=0)

    assert types.tolist() == [
        IMAGE_PAD,
        IMAGE_PAD,
        IMAGE_PAD,
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_NEW_LINE,
        IMAGE,
        IMAGE_PAD,
        IMAGE,
        IMAGE_PAD,
        IMAGE_NEW_LINE,
        IMAGE_PAD,
        IMAGE_END,
    ]
    assert permutation.tolist() == [0, 2, 1, 3, 4, 5]


def test_preprocess_image_respects_patch_and_token_budgets() -> None:
    config = _tiny_vision_config()
    image = Image.fromarray(np.zeros((40, 800, 3), dtype=np.uint8))

    prepared = preprocess_image(image, config)
    _, _, num_tokens = grid_tokens(
        prepared.n_vit_h * config.vision_patch_size,
        prepared.n_vit_w * config.vision_patch_size,
        config.vision_patch_size,
        config.vision_downsample_ratio,
    )

    assert prepared.patches.shape == (
        prepared.n_vit_h * prepared.n_vit_w,
        3,
        config.vision_patch_size,
        config.vision_patch_size,
    )
    assert prepared.patches.dtype == torch.bfloat16
    assert num_tokens + 3 <= config.vision_max_n_token


def test_tiny_vision_tower_and_aligner_preserve_official_shapes() -> None:
    config = _tiny_vision_config()
    vision = DeepseekV4VisionTransformer(config)
    aligner = DeepseekV4VisionAligner(config)
    patches = torch.randn(12, 3, 2, 2)

    vision_features = vision(patches, n_h=3, n_w=4)
    aligned = aligner(vision_features, n_h=3, n_w=4)

    assert vision_features.shape == (12, config.vision_dim)
    assert aligned.shape == (4, config.hidden_size)
    vision_names = set(vision.state_dict())
    assert "patch_embed.proj.weight" in vision_names
    assert "blocks.0.attn.wqkv.bias" in vision_names
    assert "blocks.1.mlp.w2.weight" in vision_names
    assert set(aligner.state_dict()) == {
        "w1.weight",
        "w1.bias",
        "w2.weight",
        "w2.bias",
    }


def test_position_dependent_padding_keeps_image_grid_four_aligned() -> None:
    vocab_size = 100
    image_pad_id = vocab_size + IMAGE_PAD
    types, _ = build_image_block(2, 2, start_pos=3)
    block = (types + vocab_size).tolist()
    embed_mask = torch.tensor(block) == vocab_size + IMAGE
    first_start = 1
    second_start = first_start + len(block) + 2
    token_ids = [7] + block + [8, 9] + block + [10]
    placeholders = [
        PlaceholderFeaturesInfo("image", 0, first_start, block, embed_mask),
        PlaceholderFeaturesInfo("image", 1, second_start, block, embed_mask),
    ]

    output, positioned = _position_image_blocks(
        token_ids,
        placeholders,
        image_pad_id,
    )

    assert output[-1] == 10
    for placeholder in positioned:
        start_marker = placeholder.tokens.index(vocab_size + IMAGE_START)
        assert (placeholder.start_idx + start_marker + 1) % 4 == 0
        assert placeholder.is_embed is not None
        assert int(placeholder.is_embed.sum()) == 4
        assert len(placeholder.is_embed) == len(placeholder.tokens)


def test_oov_image_types_use_official_learned_embedding_map() -> None:
    vocab_size = 100
    input_ids = torch.tensor([5, 100, 101, 102, 103, 104, 6])
    text_embeddings = torch.full((7, 3), -1.0)
    image_type_embeddings = torch.arange(15, dtype=torch.float32).view(5, 3)

    output = apply_image_type_embeddings(
        text_embeddings,
        input_ids,
        vocab_size,
        image_type_embeddings,
    )

    assert torch.equal(output[[0, 6]], text_embeddings[[0, 6]])
    assert torch.equal(output[1:6], image_type_embeddings)


def test_config_exposes_vision_fields_without_enabling_text_checkpoints() -> None:
    text_config = DeepseekV4Config()
    vision_config = DeepseekV4Config(vision_n_layers=32, vision_max_n_token=384)

    assert text_config.vision_n_layers == 0
    assert text_config.is_mm_prefix_lm is False
    assert vision_config.vision_n_layers == 32
    assert vision_config.vision_max_n_token == 384
    assert vision_config.is_mm_prefix_lm is True


def test_full_image_placeholder_and_swa_visibility_match_official_semantics() -> None:
    types, _ = build_image_block(3, 2, start_pos=0)
    assert set(types.tolist()) == {
        IMAGE_START,
        IMAGE_PAD,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    }

    placeholder = PlaceholderRange(
        offset=20,
        length=len(types),
        is_embed=types == IMAGE,
    )
    image_range = (placeholder.offset, placeholder.offset + placeholder.length - 1)

    # The legacy/default range still contains only tower embedding positions.
    assert placeholder.extract_embeds_range() != [image_range]
    # Vision-Exp opts into the complete START..END block.
    assert placeholder.extract_embeds_range(full_placeholder=True) == [image_range]

    for position in range(image_range[0], image_range[1] + 1):
        start, end = get_image_swa_bounds(position, 4, image_range, seq_len=64)
        assert start <= image_range[0]
        assert end > image_range[1]


def test_image_swa_visibility_does_not_cross_images_or_widen_text() -> None:
    window_size = 4
    first = (8, 12)
    second = (24, 28)

    # Image queries see their own complete block, but not another image.
    first_bounds = get_image_swa_bounds(10, window_size, first, seq_len=32)
    assert first_bounds == (7, 13)
    assert not (first_bounds[0] <= second[0] < first_bounds[1])

    second_bounds = get_image_swa_bounds(26, window_size, second, seq_len=32)
    assert second_bounds == (23, 29)
    assert not (second_bounds[0] <= first[1] < second_bounds[1])

    # Text between blocks remains exactly causal SWA, with no future visibility.
    assert get_image_swa_bounds(18, window_size, None, seq_len=32) == (15, 19)


def test_official_image_swa_width_is_power_of_two_for_triton() -> None:
    assert get_max_image_swa_width(window_size=128, max_image_tokens=384) == 512
    assert get_max_image_swa_width(window_size=128, max_image_tokens=0) == 128


def test_ampere_image_ranges_do_not_change_compressed_topk_selection() -> None:
    """Pin the production Triton kernel's split between causal top-k and SWA."""
    source_path = (
        Path(__file__).parents[2]
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "rocm.py"
    )
    module = ast.parse(source_path.read_text())
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_combine_topk_swa_indices_kernel"
    )
    topk_assignment = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "topk_len"
            for target in node.targets
        )
    )
    referenced_names = {
        node.id
        for node in ast.walk(topk_assignment.value)
        if isinstance(node, ast.Name)
    }

    assert referenced_names == {"COMPRESS_RATIO", "TOP_K", "pos", "tl"}


def test_weight_mapper_preserves_official_vision_and_vl_router_names() -> None:
    from vllm.models.deepseek_v4.nvidia.model import (
        _make_deepseek_v4_weights_mapper,
    )

    mapper = _make_deepseek_v4_weights_mapper("fp4")
    source_names = [
        "vision.patch_embed.proj.weight",
        "vision.blocks.0.attn.wqkv.bias",
        "aligner.w1.weight",
        "image_start",
        "layers.3.ffn.gate.bias_vl",
    ]

    assert mapper.apply_list(source_names) == [
        "vision.patch_embed.proj.weight",
        "vision.blocks.0.attn.wqkv.bias",
        "aligner.w1.weight",
        "image_start",
        "model.layers.3.ffn.gate.bias_vl",
    ]


def test_dspark_pp_keeps_a_local_copy_of_the_target_embedding() -> None:
    from vllm.models.deepseek_v4.nvidia.dspark import (
        _remap_pp_owned_dspark_weight,
        _require_pp_embedding_loaded,
    )

    assert _remap_pp_owned_dspark_weight("embed.weight", use_pp=True) == (
        "model.embed_tokens.weight"
    )
    assert _remap_pp_owned_dspark_weight("embed.weight", use_pp=False) is None
    assert _remap_pp_owned_dspark_weight("layers.0.attn.wq_a.weight", True) is None
    _require_pp_embedding_loaded({"model.embed_tokens.weight"}, use_pp=True)
    with pytest.raises(RuntimeError, match="did not load embed.weight"):
        _require_pp_embedding_loaded(set(), use_pp=True)


def test_dspark_pp_requires_aux_outputs_on_last_stage() -> None:
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
        _validate_last_pp_stage_aux_layers,
    )

    valid = SimpleNamespace(
        aux_hidden_state_layers=(41, 42, 43), start_layer=36, end_layer=43
    )
    _validate_last_pp_stage_aux_layers(valid, use_pp=True)

    split = SimpleNamespace(
        aux_hidden_state_layers=(41, 42, 43), start_layer=42, end_layer=43
    )
    with pytest.raises(RuntimeError, match="every auxiliary output"):
        _validate_last_pp_stage_aux_layers(split, use_pp=True)
