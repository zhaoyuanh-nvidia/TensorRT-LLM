# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in, CUDA-graph-safe confidence/acceptance trace collection."""

import atexit
import os
from pathlib import Path

import torch

from tensorrt_llm.logger import logger

CONFIDENCE_TRACE_PATH_ENV = "TLLM_DSPARK_CONFIDENCE_TRACE_PATH"
CONFIDENCE_TRACE_ROWS_ENV = "TLLM_DSPARK_CONFIDENCE_TRACE_ROWS"

__all__ = [
    "CONFIDENCE_TRACE_PATH_ENV",
    "ConfidenceTraceRing",
    "confidence_trace_path_from_env",
]


def confidence_trace_path_from_env() -> str | None:
    """Return the configured trace stem, or ``None`` when collection is off."""
    value = os.environ.get(CONFIDENCE_TRACE_PATH_ENV, "").strip()
    return value or None


class ConfidenceTraceRing:
    """Pair full-K confidence logits with next-step acceptance labels.

    All record/update operations are tensor-only and can be captured in the
    target CUDA graph.  A device counter supplies dynamic ring indices during
    replay, so Python is not required on the replay path.  The process writes
    one rank-local ``.pt`` shard at normal exit.
    """

    def __init__(
        self,
        *,
        path_stem: str,
        rank: int,
        num_slots: int,
        max_draft_len: int,
        scratch_slot: int,
        capacity: int | None = None,
    ) -> None:
        if capacity is None:
            capacity = int(os.environ.get(CONFIDENCE_TRACE_ROWS_ENV, "65536"))
        if capacity < num_slots:
            raise ValueError(
                f"{CONFIDENCE_TRACE_ROWS_ENV}={capacity} must be at least num_slots={num_slots}"
            )
        self.path = Path(f"{path_stem}.rank{rank}.pt")
        self.rank = rank
        self.capacity = capacity
        self.max_draft_len = max_draft_len
        self.scratch_slot = scratch_slot
        self._flushed = False

        self._slot_logits = torch.zeros(
            (num_slots, max_draft_len), dtype=torch.float32, device="cuda"
        )
        self._slot_valid = torch.zeros(num_slots, dtype=torch.bool, device="cuda")
        self._slot_graph_batch_size = torch.zeros(num_slots, dtype=torch.int32, device="cuda")
        self._slot_step_id = torch.zeros(num_slots, dtype=torch.int64, device="cuda")
        self._logits = torch.zeros((capacity, max_draft_len), dtype=torch.float32, device="cuda")
        self._prefix_mask = torch.zeros_like(self._logits)
        self._valid = torch.zeros(capacity, dtype=torch.bool, device="cuda")
        self._graph_batch_size = torch.zeros(capacity, dtype=torch.int32, device="cuda")
        self._step_id = torch.zeros(capacity, dtype=torch.int64, device="cuda")
        self._counter = torch.zeros((), dtype=torch.int64, device="cuda")
        self._event_counter = torch.zeros((), dtype=torch.int64, device="cuda")
        atexit.register(self.flush)
        logger.warning(
            f"DSpark confidence tracing enabled: capacity={capacity} rows, "
            f"output={self.path}. Collection is diagnostic and adds device memory."
        )

    def invalidate_slot(self, slot: int) -> None:
        """Prevent a recycled request slot from pairing with stale logits."""
        self._slot_valid[slot] = False

    def record_and_update(
        self,
        *,
        slots: torch.Tensor,
        confidence_logits: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        """Record the previous block and publish the current block by slot."""
        num_requests = confidence_logits.shape[0]
        if confidence_logits.shape != (num_requests, self.max_draft_len):
            raise ValueError("confidence logits do not match the trace-ring geometry")
        if slots.shape != (num_requests,) or num_accepted_tokens.shape != (num_requests,):
            raise ValueError("trace-ring slots and accepted lengths must have shape [G]")

        slots = slots.to(torch.long)
        real_rows = slots != self.scratch_slot
        previous_logits = self._slot_logits.index_select(0, slots)
        previous_valid = self._slot_valid.index_select(0, slots) & real_rows
        previous_graph_batch_size = self._slot_graph_batch_size.index_select(0, slots)
        previous_step_id = self._slot_step_id.index_select(0, slots)
        accepted_drafts = (num_accepted_tokens.to(torch.int64) - 1).clamp(
            min=0, max=self.max_draft_len
        )
        positions = torch.arange(self.max_draft_len, device=confidence_logits.device)
        prefix_mask = positions.unsqueeze(0) < accepted_drafts.unsqueeze(1)

        row_indices = torch.remainder(
            self._counter + torch.arange(num_requests, device=confidence_logits.device),
            self.capacity,
        ).to(torch.long)
        self._logits.index_copy_(0, row_indices, previous_logits)
        self._prefix_mask.index_copy_(0, row_indices, prefix_mask.to(torch.float32))
        self._valid.index_copy_(0, row_indices, previous_valid)
        self._graph_batch_size.index_copy_(0, row_indices, previous_graph_batch_size)
        self._step_id.index_copy_(0, row_indices, previous_step_id)
        self._counter.add_(num_requests)

        current = torch.where(real_rows.unsqueeze(1), confidence_logits.float(), 0.0)
        self._slot_logits.index_copy_(0, slots, current)
        self._slot_valid.index_copy_(0, slots, real_rows)
        self._slot_graph_batch_size.index_copy_(
            0,
            slots,
            torch.full_like(slots, num_requests, dtype=torch.int32),
        )
        self._slot_step_id.index_copy_(0, slots, self._event_counter.expand(num_requests))
        self._event_counter.add_(1)

    @torch.inference_mode()
    def reset(self) -> None:
        """Discard warmup/capture state while preserving captured buffer addresses."""
        if self._flushed:
            raise RuntimeError("cannot reset a flushed DSpark confidence trace ring")
        self._slot_valid.zero_()
        self._valid.zero_()
        self._counter.zero_()
        self._event_counter.zero_()
        logger.warning("DSpark confidence trace: discarded warmup and CUDA-capture rows")

    def flush(self) -> None:
        """Write valid paired rows to a CPU shard once."""
        if self._flushed:
            return
        self._flushed = True
        try:
            torch.cuda.synchronize()
            counter = int(self._counter.item())
            written = min(counter, self.capacity)
            if written == 0:
                logger.warning("DSpark confidence trace ring is empty; no shard written")
                return
            if counter <= self.capacity:
                order = torch.arange(written, device="cuda")
            else:
                start = counter % self.capacity
                order = torch.cat(
                    (
                        torch.arange(start, self.capacity, device="cuda"),
                        torch.arange(0, start, device="cuda"),
                    )
                )
            valid = self._valid.index_select(0, order)
            order = order[valid]
            payload = {
                "pairing": "draft_seq_ring",
                "logits": self._logits.index_select(0, order).cpu(),
                "prefix_mask": self._prefix_mask.index_select(0, order).cpu(),
                "graph_batch_size": self._graph_batch_size.index_select(0, order).cpu(),
                "step_id": self._step_id.index_select(0, order).cpu(),
                "max_draft_len": self.max_draft_len,
                "rank": self.rank,
                "rows_seen": counter,
                "rows_written": int(order.numel()),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.path)
            logger.warning(
                f"DSpark confidence trace: wrote {payload['rows_written']} paired "
                f"rows to {self.path}"
            )
        except (OSError, RuntimeError) as error:
            logger.error(f"Failed to flush DSpark confidence trace {self.path}: {error}")
