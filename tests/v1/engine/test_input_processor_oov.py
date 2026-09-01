# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.inputs import tokens_input
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_processor(*, vision_n_layers: int) -> InputProcessor:
    vocab_size = 100
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        max_model_len=16,
        runner_type="generate",
        hf_config=SimpleNamespace(vision_n_layers=vision_n_layers),
        get_vocab_size=lambda: vocab_size,
    )
    processor.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(max_token_id=vocab_size - 1)
    )
    processor.skip_prompt_length_check = False
    processor.supports_mm_inputs = False
    processor.mm_encoder_cache_size = 0
    return processor


def _validate_token(processor: InputProcessor, token_id: int) -> None:
    processor._validate_model_input(tokens_input([token_id]), "decoder")


def test_vision_accepts_last_image_sentinel() -> None:
    _validate_token(_make_processor(vision_n_layers=32), 104)


def test_vision_rejects_token_above_image_sentinels() -> None:
    with pytest.raises(VLLMValidationError, match="Token id 105"):
        _validate_token(_make_processor(vision_n_layers=32), 105)


def test_text_model_rejects_token_at_vocab_size() -> None:
    with pytest.raises(VLLMValidationError, match="Token id 100"):
        _validate_token(_make_processor(vision_n_layers=0), 100)


def test_prompt_logprobs_reject_oov_image_sentinel() -> None:
    processor = _make_processor(vision_n_layers=32)
    params = SamplingParams(prompt_logprobs=1)

    with pytest.raises(VLLMValidationError, match="prompt_logprobs"):
        processor._validate_prompt_logprobs(params, [104])
