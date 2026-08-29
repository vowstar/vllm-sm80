# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention with split-KV for low-batch decode."""

# Ported from an sm_80-proven internal A100 fork and extended with a
# NoPE variant for GLM-5.3-Flash (qk_rope_head_dim=0): the per-head
# qk row is a bare
# kv_lora_rank=512 with no 64-lane RoPE tail. `BLOCK_DPE` was already a
# constexpr kernel parameter; it is now derived per launch from the q/kv
# row width (576 -> 64, 512 -> 0) and every reference to the PE lanes is
# guarded by `BLOCK_DPE > 0` so the 0-width case compiles. Upstream's FA3
# sparse backend handles the same NoPE shape with a dummy-64 + only_qv
# trick; making BLOCK_DPE a real 0 here avoids the 12.5% dummy bandwidth.
#
# FP8 KV (e4m3fn, per-tensor scale): Triton on sm_80 rejects fp8e4nv
# pointer element types ("not supported in this architecture"), so the
# cache stays uint8-typed and every load is dequantized in-kernel with
# a 6-op bit assembly into fp16 (e4m3 values are exact in fp16). The
# dot then runs on the fp16 tensor-core path; no fp8 MMA (sm_89+) is
# needed. Measured on CMP 170HX (sm_80): uint8 gather + assembly keeps
# ~0.9x of the bf16 kernel's effective bandwidth while reading half
# the bytes; heavier dequant schemes (fp32/uint16 bit math, 256-entry
# LUT) collapse to ~0.3x.

import functools

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units

# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
_BLOCK_DMODEL = 512
_BLOCK_DV = 512
# Supported per-head qk row widths: 576 = 512 + 64 RoPE (DeepSeek),
# 512 = 512 + 0 (GLM-5.3 NoPE). _DIM_QK kept for the DeepSeek default.
_BLOCK_DPE = 64
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576
_DIM_QK_NOPE = _BLOCK_DMODEL  # 512, GLM-5.3 NoPE

_BLOCK_H = 16

# Merge kernel grid is spread across heads and DV tiles to avoid a (1,1)
# launch starving the SMs (pattern from FlashMLA's combine kernel).
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Final (prefill) and split (decode) kernels each tune to their own regime.
_FINAL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=nw, num_stages=ns)
    for nw in (2, 4)
    for ns in (2, 4)
]
_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Smallest BLOCK_N the sweep offers; the topk-divisibility check at
# dispatch time keeps every tile full.
_MIN_BLOCK_N = min(
    c.kwargs["BLOCK_N"] for c in _FINAL_AUTOTUNE_CONFIGS + _SPLIT_AUTOTUNE_CONFIGS
)

# Split-count candidates for `_choose_num_kv_splits`; also the set pre-compiled
# by `_warmup_autotune`.
KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)

_MIN_TOPK_PER_SPLIT = 128  # below this, per-split work is too small to amortize
_SPLIT_MAX_OCCUPANCY = 4  # skip split when baseline grid fills >=1/4 of SMs


@triton.jit
def _dequant_fp8_e4m3(b):
    """uint8 e4m3fn bit pattern -> fp16, WITHOUT the per-tensor scale.

    ((b&0x80)<<8) | (((b&0x7F)<<7) + 0x2000): sign to bit 15, exponent
    rebased +8 (e4m3 bias 7 -> fp16 bias 15), mantissa to the fp16 top
    bits. Exact for normals and every e4m3 value stays exact in fp16,
    which also puts the dot on the fp16 tensor-core path. Subnormal
    inputs (true |x| < 2^-6) read as (1+m/8)*2^-7 instead of m*2^-9:
    an absolute error below 0.016 on the ~1% of RMS-normalized latents
    that small, far below the attention noise floor. The per-tensor
    scale is applied separately as scalar multiplies (see call sites).
    """
    bi = b.to(tl.uint16)
    h = ((bi & 0x80) << 8) | (((bi & 0x7F) << 7) + 0x2000)
    return h.to(tl.float16, bitcast=True)


@triton.jit
def _sparse_mla_compute_tile(
    q_buffer,
    k_buffer,  # V is the first BLOCK_DV lanes of each row of k_buffer.
    indices_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    kv_scale_ptr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    """Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators."""
    if KV_IS_FP8:
        kv_scale = tl.load(kv_scale_ptr)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    if KV_IS_FP8:
        # fp16 compute path for the fp8 cache: q converts once per
        # program instead of converting every gathered K/V tile to bf16.
        q = q.to(tl.float16)
    # NoPE variant — the RoPE lanes only
    # exist when BLOCK_DPE > 0 (tl.arange(0, 0) is illegal, and the extra
    # dot would read past the 512-wide NoPE row).
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        qpe = tl.load(
            q_buffer
            + cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        if KV_IS_FP8:
            qpe = qpe.to(tl.float16)

    # Finite sentinel (not -inf) — when an entire BLOCK_N tile is masked,
    # `-inf - -inf = NaN` poisons the softmax; `sentinel - sentinel = 0`
    # gives `exp2(0) = 1` and the matching V rows are already 0.
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        )
        mask_kv = (indices >= 0) & (indices < seq_kv)

        # widen address math to int64.
        # With a 918-block x 4608-row pool and stride_kv_token=512, rows
        # >= 2**31/512 (= 4,194,304, i.e. physical blocks 910+) overflow
        # int32 and wrap negative — they pass mask_kv (which tests the
        # index, not the offset) and load wild addresses (rank0 IMA in
        # production on a dirty pool). Same latent bug exists upstream
        # (#38476/#47629 kernels are arithmetically identical).
        idx64 = indices.to(tl.int64)

        offs_k = (
            idx64[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        if KV_IS_FP8:
            k = _dequant_fp8_e4m3(k)
        qk = tl.dot(q, k.to(q.dtype))

        # NoPE variant — skip the PE dot
        # entirely when BLOCK_DPE == 0 (see above).
        if BLOCK_DPE > 0:
            offs_kpe = (
                idx64[None, :] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=mask_kv[None, :],
                other=0.0,
            )
            if KV_IS_FP8:
                kpe = _dequant_fp8_e4m3(kpe)
            qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        if KV_IS_FP8:
            # K-side dequant scale: one scalar broadcast on the fp32
            # logits instead of a per-element multiply on the K tile.
            qk *= kv_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        offs_v = (
            idx64[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)
        if KV_IS_FP8:
            v = _dequant_fp8_e4m3(v)
            # The assembly maps the masked-load 0x00 filler to ~0.008,
            # not 0; fully-masked tiles rely on zero V rows (see the
            # NEG_LARGE sentinel above), so re-apply the mask.
            v = tl.where(mask_kv[:, None], v, 0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    if KV_IS_FP8:
        # V-side dequant scale: fold into the accumulator once per
        # program instead of a per-element multiply on the V tile.
        acc *= kv_scale
    return acc, e_max, e_sum


@triton.autotune(configs=_FINAL_AUTOTUNE_CONFIGS, key=["index_topk", "kv_group_num"])
@triton.jit
def _sparse_mla_kernel_final(
    q_buffer,
    k_buffer,
    indices_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    kv_scale_ptr,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, _, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        kv_scale_ptr,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        KV_IS_FP8,
    )

    # Guard against queries with zero valid KV (e_sum == 0 → NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=["index_topk", "NUM_KV_SPLITS", "kv_group_num"],
)
@triton.jit
def _sparse_mla_kernel_split(
    q_buffer,
    k_buffer,
    indices_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    kv_scale_ptr,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    """Stage 1 of split-KV: process one slice of the topk axis and write
    its `(out_partial, lse_partial)` into the mid buffer."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        kv_scale_ptr,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        KV_IS_FP8,
    )

    # Partial output and natural-log LSE for stage-2 merge.
    # When a split has no valid KV (`e_sum == 0`), guard the divide so the
    # mid buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term
    # in stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(
        mid_base_2d + offs_dv[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@triton.jit
def _sparse_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
):
    """Stage 2: N-way online-softmax merge of per-split `(out, lse)` tiles.

    Grid is `(num_tokens, num_head_groups, num_dv_tiles)`. Each program handles
    `BLOCK_H` heads × `BLOCK_DV_TILE` output-dim lanes. The LSE reduction is
    identical across DV tiles for the same (token, head) — each program
    recomputes it locally, which is cheap (O(NUM_KV_SPLITS) scalars) and
    avoids inter-CTA synchronization.
    """
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    # Finite sentinel — same NaN guard as the split kernel for empty splits.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, index_topk: int, sm_count: int
) -> int:
    """Pick a power-of-2 split count that fills the device without dropping
    per-split work below _MIN_TOPK_PER_SPLIT. Returns 1 when the single-pass
    grid already reaches ~1/_SPLIT_MAX_OCCUPANCY utilization.
    """
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)  # floor to power of 2
    num_kv_splits = min(ideal, max_splits)
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


def triton_mla_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over topk indices.

    Args:
        q:         [num_tokens, num_heads_q, dim_qk] bf16
        kv:        [seq_kv, num_heads_kv=1, dim_qk] bf16, or uint8 holding
                   e4m3fn bit patterns when `kv_scale` is given
        indices:   [num_tokens, num_heads_kv=1, topk] int32
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = force single-pass.
        sm_count:  cached device SM count for the split heuristic.
        kv_scale:  per-tensor fp32 dequant scale for an fp8 (uint8) cache;
                   None means the cache is bf16.

    Returns:
        out:   [num_tokens, num_heads_q, _BLOCK_DV] bf16
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    # NoPE variant — accept the GLM-5.3
    # 512-wide row (BLOCK_DPE=0) alongside the DeepSeek 576 (BLOCK_DPE=64).
    assert dim_qk in (_DIM_QK, _DIM_QK_NOPE), (
        f"sparse MLA kernel requires dim_qk in ({_DIM_QK}, {_DIM_QK_NOPE}) "
        f"(DeepSeek-V3.2 RoPE / GLM-5.3 NoPE), got {dim_qk}"
    )
    block_dpe = dim_qk - _BLOCK_DMODEL
    assert kv.shape[1] == 1 and kv.shape[2] == dim_qk
    index_topk = indices.shape[2]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of the smallest autotune "
        f"BLOCK_N ({_MIN_BLOCK_N})"
    )

    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, min(_BLOCK_H, kv_group_num))

    kv_is_fp8 = kv_scale is not None
    if kv_is_fp8:
        assert kv.dtype == torch.uint8, (
            f"fp8 KV cache must reach the kernel as uint8 bytes, got {kv.dtype}"
        )
        assert kv_scale.dtype == torch.float32 and kv_scale.numel() == 1
    else:
        # Signature placeholder; dead code under KV_IS_FP8=False.
        kv_scale = q

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = num_compute_units(q.device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens, num_head_groups, index_topk, sm_count
        )

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )

    if num_kv_splits == 1:
        _sparse_mla_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            k_buffer=kv,
            indices_ptr=indices,
            out_ptr=out,
            seq_kv=kv.shape[0],
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            kv_scale_ptr=kv_scale,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=_BLOCK_H,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=block_dpe,
            KV_IS_FP8=kv_is_fp8,
        )
        return out

    # Split-KV: partial fp32 output + LSE per (token, head, split).
    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        k_buffer=kv,
        indices_ptr=indices,
        mid_out_ptr=mid_out,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_kv_token=kv.stride(0),
        stride_kv_head=kv.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        kv_scale_ptr=kv_scale,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        BLOCK_DPE=block_dpe,
        LOGE2=LOGE2,
        KV_IS_FP8=kv_is_fp8,
    )

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out
