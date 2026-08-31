# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-modal processor for DeepSeek V4 Vision.

The image transform and token layout follow the official model repository at
revision ``e46e16bf6035c6f317eb2ac7458eb0362926d402``.
"""

import math
from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult,
    PlaceholderFeaturesInfo,
)

from .vision import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_PAD,
    build_image_block,
    grid_tokens,
    preprocess_image,
)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def _has_vision(config: object) -> bool:
    return int(getattr(config, "vision_n_layers", 0)) > 0


def _position_image_blocks(
    token_ids: list[int],
    placeholders: Sequence[PlaceholderFeaturesInfo],
    image_pad_token_id: int,
) -> tuple[list[int], list[PlaceholderFeaturesInfo]]:
    """Insert position-dependent compression padding before image blocks."""
    output: list[int] = []
    output_placeholders: list[PlaceholderFeaturesInfo] = []
    prev_end = 0
    for placeholder in placeholders:
        output.extend(token_ids[prev_end : placeholder.start_idx])
        start_idx = len(output)
        num_padding = COMPRESS_PAD_TO - 1 - start_idx % COMPRESS_PAD_TO
        tokens = [image_pad_token_id] * num_padding + placeholder.tokens
        output.extend(tokens)

        is_embed = placeholder.is_embed
        if is_embed is not None:
            is_embed = torch.cat([torch.zeros(num_padding, dtype=torch.bool), is_embed])
        output_placeholders.append(
            PlaceholderFeaturesInfo(
                modality=placeholder.modality,
                item_idx=placeholder.item_idx,
                start_idx=start_idx,
                tokens=tokens,
                is_embed=is_embed,
            )
        )
        prev_end = placeholder.start_idx + placeholder.length

    output.extend(token_ids[prev_end:])
    return output, output_placeholders


class DeepseekV4VisionProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        if not _has_vision(self.get_hf_config()):
            return {}
        return {"image": None}

    @property
    def image_placeholder_id(self) -> int:
        tokenizer = self.get_tokenizer()
        token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return int(token_id)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"image": int(self.get_hf_config().vision_max_n_token)}

    def get_image_size_with_most_features(self) -> ImageSize:
        config = self.get_hf_config()
        token_budget = int(config.vision_max_n_token) - (COMPRESS_PAD_TO - 1)
        max_wh_ratio = getattr(config, "vision_max_wh_ratio", None)
        best_area = 0
        best_grid = (1, 1)
        for n_llm_h in range(1, token_budget + 1):
            for n_llm_w in range(1, token_budget + 1):
                if max_wh_ratio is not None and n_llm_w > n_llm_h * max_wh_ratio:
                    continue
                _, _, num_tokens = grid_tokens(n_llm_h, n_llm_w, 1, 1)
                area = n_llm_h * n_llm_w
                if num_tokens <= token_budget and area > best_area:
                    best_area = area
                    best_grid = (n_llm_h, n_llm_w)

        patch_stride = int(config.vision_patch_size) * int(
            config.vision_downsample_ratio
        )
        return ImageSize(
            width=best_grid[1] * patch_stride,
            height=best_grid[0] * patch_stride,
        )


class DeepseekV4VisionDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        width, height = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=width,
                height=height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class DeepseekV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VisionProcessingInfo]
):
    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del hf_processor_mm_kwargs
        valid_items = mm_items.select(
            {key for key, count in mm_items.get_all_counts().items() if count > 0}
        )
        mm_data, passthrough_data = self._get_hf_mm_data(valid_items)
        images = mm_data.get("images", ())
        if not isinstance(images, Sequence):
            raise TypeError("DeepSeek V4 image inputs must be a sequence")

        config = self.info.get_hf_config()
        pixel_values: list[torch.Tensor] = []
        image_grid_hws: list[list[int]] = []
        for image in images:
            prepared = preprocess_image(image, config)
            pixel_values.append(prepared.patches)
            image_grid_hws.append([prepared.n_vit_h, prepared.n_vit_w])

        prompt_ids = self.info.get_tokenizer().encode(
            self.dummy_inputs.get_dummy_text(mm_items.get_all_counts()),
            add_special_tokens=False,
        )
        data: dict[str, object] = {"input_ids": [prompt_ids]}
        if pixel_values:
            data.update(
                pixel_values=pixel_values,
                image_grid_hws=torch.tensor(image_grid_hws, dtype=torch.int64),
            )
        processed = BatchFeature(data=data, tensor_type=None)
        processed.update(passthrough_data)
        return processed

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        if "pixel_values" not in hf_inputs:
            return {}
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_grid_hws": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs
        config = self.info.get_hf_config()
        placeholder_id = self.info.image_placeholder_id
        image_token_id = int(config.vocab_size) + IMAGE

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            grid = out_mm_kwargs["image"][item_idx]["image_grid_hws"].data
            n_vit_h, n_vit_w = (int(value) for value in grid.reshape(-1).tolist())
            ratio = int(config.vision_downsample_ratio)
            n_llm_h = math.ceil(n_vit_h / ratio)
            n_llm_w = math.ceil(n_vit_w / ratio)
            types, _ = build_image_block(n_llm_h, n_llm_w, start_pos=3)
            tokens = (types + int(config.vocab_size)).tolist()
            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[placeholder_id],
                replacement=get_replacement,
            )
        ]

    def _apply_token_matches_with_placeholders(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[
        list[int],
        MultiModalPromptUpdatesApplyResult,
        Mapping[str, list[PlaceholderFeaturesInfo]],
    ]:
        new_token_ids, result, placeholders = (
            super()._apply_token_matches_with_placeholders(
                token_ids,
                mm_prompt_updates,
            )
        )
        image_placeholders = placeholders.get("image", [])
        if not image_placeholders:
            return new_token_ids, result, placeholders

        config = self.info.get_hf_config()
        image_pad_token_id = int(config.vocab_size) + IMAGE_PAD
        new_token_ids, image_placeholders = _position_image_blocks(
            new_token_ids,
            image_placeholders,
            image_pad_token_id,
        )
        return new_token_ids, result, {"image": image_placeholders}
