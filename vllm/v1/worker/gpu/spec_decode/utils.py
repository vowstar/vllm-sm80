# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    """Step-tagged ring of draft-token snapshots (vllm#54437 part 2).

    The old implementation kept a single slot that every execute_model
    overwrote. Under PP + async scheduling the deferred-sampling branch then
    back-filled a step with whatever batch ran last, which is a *different*
    microbatch: structured-output requests silently lost their drafts (or
    worse, received another step's real tokens) and grammar_bitmask advanced
    along the wrong window. Snapshots are now kept per scheduler step and the
    engine asks for the step it is sampling by equality; a miss returns None
    and the caller falls back to the fail-closed path (placeholders survive,
    drafts invalidated by num_acceptable_drafts).
    """

    def __init__(self, device: torch.device | None = None, max_snapshots: int = 16):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.max_snapshots = max_snapshots

        # step_id -> (req_ids, draft_tokens_np, num_draft_tokens, copy_event)
        # draft_tokens_np is None for batches without structured-output
        # requests (validation not needed, get returns all -1).
        self.snapshots: OrderedDict[
            int, tuple[list[str], np.ndarray | None, int, torch.cuda.Event]
        ] = OrderedDict()
        # Legacy single-slot view (post_step / non-async path).
        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self.copy_event = torch.cuda.Event(blocking=True)

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        step_id: int = 0,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        draft_tokens_np: np.ndarray | None = None
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
        else:
            # For spec decoding + structured outputs, we must transfer the
            # draft tokens back to the scheduler for grammar validation.
            current_stream = torch.cuda.current_stream(self.device)
            self.copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.copy_stream):
                draft_tokens_np = async_copy_to_np(draft_tokens)
                # draft_tokens is a temporary allocation on the main stream and read here on
                # copy_stream; without record_stream, the caching allocator may reuse its
                # memory before the async copy executes.
                draft_tokens.record_stream(self.copy_stream)
                self.copy_event.record()
            self.draft_tokens_np = draft_tokens_np

        # Per-step copy event: copies run in order on copy_stream, so syncing
        # a later event covers every earlier snapshot as well. Reuse the
        # shared event for the latest snapshot; earlier snapshots keep their
        # own event object.
        event = (
            self.copy_event
            if input_batch.has_structured_output_reqs
            else torch.cuda.Event(blocking=True)
        )
        self.snapshots[step_id] = (
            list(input_batch.req_ids),
            draft_tokens_np,
            self.num_draft_tokens,
            event,
        )
        self.snapshots.move_to_end(step_id)
        while len(self.snapshots) > self.max_snapshots:
            self.snapshots.popitem(last=False)

    def get_draft_tokens(self, step_id: int | None = None) -> DraftTokenIds | None:
        if step_id is not None:
            snap = self.snapshots.get(step_id)
            if snap is None:
                # Identity miss: the caller leaves the step's placeholders in
                # place, which the fail-closed path treats as an invalid
                # window. Conservative and correct.
                return None
            req_ids, draft_tokens_np, num_draft_tokens, event = snap
            if draft_tokens_np is not None:
                event.synchronize()
                draft_token_ids = draft_tokens_np.tolist()
            else:
                draft_token_ids = [[-1] * num_draft_tokens for _ in req_ids]
            return DraftTokenIds(req_ids, draft_token_ids)

        # Legacy path (post_step / non-async scheduling): latest batch.
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
