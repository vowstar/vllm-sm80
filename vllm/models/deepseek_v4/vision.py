# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision encoder and image layout for DeepSeek V4 Vision.

This follows the official ``DeepSeek-V4-Flash-Vision-Exp`` implementation at
revision ``e46e16bf6035c6f317eb2ac7458eb0362926d402``.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
NUM_IMAGE_TOKEN_TYPES = 5
COMPRESS_PAD_TO = 4


@dataclass
class DeepseekV4ImagePatches:
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int


def grid_tokens(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    """Return the aligned grid and full N-layout token count."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = (max_n_token - 2) // max_h - 1
        if max_w <= 1:
            raise ValueError("DeepSeek V4 image token budget is too small")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size

    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height,
            width,
            patch_size,
            downsample_ratio,
            budget,
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def preprocess_image(
    image: Image.Image | np.ndarray, config: Any
) -> DeepseekV4ImagePatches:
    """Resize and patchify one image using the official preprocessing rules."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image or NumPy array, got {type(image)!r}")

    image = image.convert("RGB")
    patch_size = int(config.vision_patch_size)
    width, height = image.size
    max_wh_ratio = getattr(config, "vision_max_wh_ratio", None)
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = int(height * max_wh_ratio)

    min_pixels = int(getattr(config, "vision_min_pixels", 0))
    if 0 < width * height < min_pixels:
        ratio = math.sqrt(min_pixels / (width * height))
        width = int(width * ratio)
        height = int(height * ratio)

    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        int(config.vision_downsample_ratio),
        int(config.vision_max_n_token),
    )
    n_vit_h = best_height // patch_size
    n_vit_w = best_width // patch_size

    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(
            image,
            (best_width, best_height),
            color=(127, 127, 127),
        )

    pixels = torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
    pixels = pixels.permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        pixels.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return DeepseekV4ImagePatches(
        patches=patches,
        n_vit_h=n_vit_h,
        n_vit_w=n_vit_w,
        n_llm_h=n_llm_h,
        n_llm_w=n_llm_w,
    )


def build_image_block(
    n_llm_h: int,
    n_llm_w: int,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the official N-layout token types and aligner permutation."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def apply_image_type_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
    image_type_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Replace OOV image token types with their learned embeddings."""
    if image_type_embeddings.shape[0] != NUM_IMAGE_TOKEN_TYPES:
        raise ValueError("DeepSeek V4 requires five image type embeddings")
    token_types = input_ids - vocab_size
    is_image_type = (token_types >= 0) & (token_types < NUM_IMAGE_TOKEN_TYPES)
    safe_token_types = token_types.clamp(0, NUM_IMAGE_TOKEN_TYPES - 1)
    return torch.where(
        is_image_type.unsqueeze(-1),
        image_type_embeddings[safe_token_types.to(torch.long)],
        inputs_embeds,
    )


@lru_cache(maxsize=8)
def _get_vision_cos_sin_cpu(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def get_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _get_vision_cos_sin_cpu(n_h, n_w, dim, theta)
    return cos.to(device=device), sin.to(device=device)


def apply_vision_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.eps
        )
        return (self.weight * normalized).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        patch_size = int(config.vision_patch_size)
        self.proj = nn.Linear(3 * patch_size**2, int(config.vision_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        vision_dim = int(config.vision_dim)
        self.n_heads = int(config.vision_n_heads)
        self.head_dim = vision_dim // self.n_heads
        self.wqkv = nn.Linear(vision_dim, 3 * vision_dim)
        self.wo = nn.Linear(vision_dim, vision_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        q, k, v = (
            tensor.view(num_tokens, self.n_heads, self.head_dim)
            for tensor in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_vision_rotary(q, cos, sin)
        k = apply_vision_rotary(k, cos, sin)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
        )
        return self.wo(output.transpose(0, 1).reshape(num_tokens, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        vision_dim = int(config.vision_dim)
        inter_dim = int(config.vision_inter_dim)
        self.w1 = nn.Linear(vision_dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        vision_dim = int(config.vision_dim)
        self.norm1 = DeepseekV4VisionRMSNorm(vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4VisionRMSNorm(vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionTransformer(nn.Module):
    """Full bidirectional ViT with DeepSeek's two-dimensional RoPE."""

    def __init__(self, config: Any):
        super().__init__()
        head_dim = int(config.vision_dim) // int(config.vision_n_heads)
        self.rope_dim = head_dim // 2
        self.rope_theta = float(config.vision_rope_theta)
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(config) for _ in range(int(config.vision_n_layers))]
        )
        self.norm = DeepseekV4VisionRMSNorm(int(config.vision_dim))

    def forward(
        self,
        patches: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h,
            n_w,
            self.rope_dim,
            self.rope_theta,
            x.device,
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4VisionAligner(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.downsample_ratio = int(config.vision_downsample_ratio)
        in_dim = int(config.vision_dim) * self.downsample_ratio**2
        hidden_size = int(config.hidden_size)
        self.w1 = nn.Linear(in_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio)
        x = x.squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
