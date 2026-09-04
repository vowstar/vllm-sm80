# SPDX-License-Identifier: Apache-2.0
"""Compile the mamba accepted-token kernels before the first request.

MambaHybrid.commit_step runs on every sampling step. Under pipeline
parallelism that call sits between two NCCL transfers, so a rank that JIT
compiles there stalls while its peers wait in the collective and the server
deadlocks instead of merely being slow. This is the same failure shape that
post_update_warmup exists for.

Two kernels reach that path. commit_step picks _scatter_num_accepted_kernel
when num_sampled is a tensor and _fill_num_accepted_kernel when it is an int,
so a deployment can meet either one first. Triton also specializes the int
argument of the fill kernel on its value, so warming a single value is not
enough; the values that actually occur are 1 and the speculative step count.

Found by driving 32 concurrent streams at a cold engine. A ladder that walks
1, 4, 8, 16 and then 32 hides this, because the kernel compiles at a low
concurrency where no collective is waiting on it.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def mamba_accepted_warmup(model_runner) -> None:
    """Best effort. A failure here must never stop a server from starting."""
    try:
        _mamba_accepted_warmup(model_runner)
    except Exception:
        logger.warning(
            "mamba accepted-token warmup did not run. The server still "
            "starts, but the commit_step kernels can JIT on the first "
            "request.",
            exc_info=True,
        )


def _collect_states(model_runner) -> list:
    """Every model-state object that owns a num_accepted_tokens_gpu buffer.

    Duck-typed on the buffer rather than on a class name. The class is
    MambaHybridModelState today, but the runner holds exactly one state object
    under .model_state and a rename upstream must not silently disable this
    warmup -- an ImportError here is caught and the kernels then JIT in a
    collective, which is the failure this module exists to prevent.
    """
    found: list = []
    seen: set[int] = set()

    def visit(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if getattr(obj, "num_accepted_tokens_gpu", None) is not None:
            found.append(obj)

    for attr in ("model_state", "model_states"):
        holder = getattr(model_runner, attr, None)
        if holder is None:
            continue
        if isinstance(holder, dict):
            for value in holder.values():
                visit(value)
        elif isinstance(holder, (list, tuple)):
            for value in holder:
                visit(value)
        else:
            visit(holder)
    return found


def _mamba_accepted_warmup(model_runner) -> None:
    states = _collect_states(model_runner)
    if not states:
        return

    from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
        _fill_num_accepted_kernel,
        _scatter_num_accepted_kernel,
    )

    device = model_runner.device
    max_num_reqs = model_runner.scheduler_config.max_num_seqs
    num_spec_steps = getattr(model_runner, "num_speculative_steps", 0) or 0

    # The ladder gpu/warmup.py walks, plus the cap itself, smallest first.
    sizes: list[int] = []
    size = 1
    while size < max_num_reqs:
        sizes.append(size)
        size *= 2
    sizes.append(max_num_reqs)

    # Triton keys an int argument on its value, so warm every value that
    # commit_step can pass: the neutral 1 and one token per speculative step.
    fill_values = sorted({1, max(num_spec_steps + 1, 1)})

    warmed = 0
    for state in states:
        target = getattr(state, "num_accepted_tokens_gpu", None)
        if target is None:
            continue
        rows = int(target.shape[0])
        for size in sizes:
            n = min(size, rows)
            if n <= 0:
                continue
            # -1 rows are the filtered-request sentinel the kernels skip; keep
            # one in the batch so that branch is compiled too.
            idx_mapping = torch.arange(n, device=device, dtype=torch.int32)
            if n > 1:
                idx_mapping[-1] = -1
            for value in fill_values:
                _fill_num_accepted_kernel[(n,)](idx_mapping, target, value)
                warmed += 1
            num_sampled = torch.ones(n, device=device, dtype=torch.int32)
            _scatter_num_accepted_kernel[(n,)](idx_mapping, num_sampled, target)
            warmed += 1

    if warmed:
        torch.cuda.synchronize()
        logger.info(
            "mamba accepted-token warmup compiled %d kernel launches over "
            "%d state objects",
            warmed,
            len(states),
        )
