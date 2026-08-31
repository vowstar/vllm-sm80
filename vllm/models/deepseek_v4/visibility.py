# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference helpers for DeepSeek V4 image-local SWA visibility."""


def get_max_image_swa_width(window_size: int, max_image_tokens: int) -> int:
    """Return a Triton-compatible width for causal SWA plus one image block."""
    width = window_size + max_image_tokens
    if width <= 0:
        raise ValueError("SWA visibility width must be positive")
    return 1 << (width - 1).bit_length()


def get_image_swa_bounds(
    position: int,
    window_size: int,
    image_range: tuple[int, int] | None = None,
    *,
    seq_len: int | None = None,
) -> tuple[int, int]:
    """Return the inclusive/exclusive SWA bounds used by Vision-Exp prefill.

    The causal window remains unchanged outside an image block. Inside one
    complete ``[IMAGE_START, IMAGE_END]`` block, visibility is the union of the
    causal window and that image block. The compressed/top-k branch is not part
    of this helper and remains causal.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if position < 0:
        raise ValueError("position must be non-negative")

    start = max(position - window_size + 1, 0)
    end = position + 1
    if image_range is not None:
        image_start, image_end = image_range
        if image_start <= position <= image_end:
            start = min(start, image_start)
            end = max(end, image_end + 1)
    if seq_len is not None:
        end = min(end, seq_len)
    return start, end
