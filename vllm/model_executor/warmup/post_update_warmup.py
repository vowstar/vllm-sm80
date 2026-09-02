# SPDX-License-Identifier: Apache-2.0
"""Compile _post_update_kernel before the first request reaches it.

post_update runs on every sampling step, and on a pipeline-parallel
deployment it also runs inside update_pp_decode_requests, which sits between
two NCCL transfers. A rank that JIT-compiles there stalls while its peers wait
in the collective, and the server deadlocks rather than merely stalling. The
upstream report of the same failure shape is vllm-project/vllm#45198, and
vllm-project/vllm#45245 fixes two other kernels the same way, by compiling
them during warmup instead of on the first request.

Warming the batch size alone is not enough for this kernel. Triton specializes
on whether a pointer argument is None, and postprocess_sampled passes
output_bin_counts only on the last pipeline rank and query_start_loc only on
some paths. Each combination is a separate compiled kernel, so this module
walks all of them.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def post_update_warmup(model_runner) -> None:
    """Best effort. A failure here must never stop a server from starting."""
    try:
        _post_update_warmup(model_runner)
    except Exception:
        logger.warning(
            "post_update warmup did not run. The server still starts, but "
            "_post_update_kernel can JIT on the first request.",
            exc_info=True,
        )


def _post_update_warmup(model_runner) -> None:
    req_states = getattr(model_runner, "req_states", None)
    if req_states is None:
        return

    from vllm.v1.worker.gpu.input_batch import post_update

    device = model_runner.device
    max_num_reqs = model_runner.scheduler_config.max_num_seqs
    num_spec_steps = getattr(model_runner, "num_speculative_steps", 0) or 0

    # The same ladder gpu/warmup.py uses, plus the cap itself, smallest first.
    sizes: list[int] = []
    size = 1
    while size < max_num_reqs:
        sizes.append(size)
        size *= 2
    sizes.append(max_num_reqs)

    bin_counts = None
    sampler = getattr(model_runner, "sampler", None)
    penalties_state = getattr(sampler, "penalties_state", None)
    if penalties_state is not None:
        bin_counts = getattr(penalties_state, "output_bin_counts", None)

    token_dtype = req_states.last_sampled_tokens.dtype
    compiled = 0
    for num_reqs in sizes:
        sampled_tokens = torch.zeros(
            (num_reqs, num_spec_steps + 1), dtype=token_dtype, device=device
        )

        def _counter(length: int, offset: int) -> torch.Tensor:
            # offset 0 gives a 16-byte aligned pointer, offset 1 does not,
            # because these are 4-byte elements. Triton specializes on that
            # alignment, so the two are different compiled kernels. The real
            # per-step counters are slices of larger buffers and land on both.
            buf = torch.zeros(length + 4, dtype=torch.int32, device=device)
            return buf[offset : offset + length]

        bin_count_variants = [None]
        if bin_counts is not None:
            bin_count_variants.append(bin_counts)

        # idx_mapping arrives as int32 on one path and int64 on another, and
        # Triton treats the two as different kernels. Measured signatures on a
        # PP4 boot showed warmup covering only int32 while the real decode path
        # used int64, so two of the four (output_bin_counts, query_start_loc)
        # combinations for int64 were never compiled and JIT-ed on the first
        # request instead.
        for idx_dtype, align_off in (
            (torch.int32, 0),
            (torch.int64, 0),
            (torch.int32, 1),
            (torch.int64, 1),
        ):
            num_sampled = _counter(num_reqs, align_off)
            num_rejected = _counter(num_reqs, align_off)
            query_start_loc = _counter(num_reqs + 1, align_off)
            # Every entry is -1, which the kernel reads as skip, so it writes
            # nothing. Only the compile is wanted here.
            idx_mapping = torch.full(
                (num_reqs,), -1, dtype=idx_dtype, device=device
            )
            for bin_count_arg in bin_count_variants:
                for qsl_arg in (None, query_start_loc):
                    post_update(
                        idx_mapping,
                        req_states.num_computed_tokens.gpu,
                        req_states.last_sampled_tokens,
                        bin_count_arg,
                        sampled_tokens,
                        num_sampled,
                        num_rejected,
                        qsl_arg,
                        req_states.all_token_ids.gpu,
                        req_states.total_len.gpu,
                    )
                    compiled += 1

    torch.cuda.synchronize()
    logger.info(
        "post_update warmup ran %d launches over %d batch sizes.",
        compiled,
        len(sizes),
    )
