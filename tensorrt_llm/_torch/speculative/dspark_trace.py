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
DYNAMIC_CONFIDENCE_TRACE_PATH_ENV = "TLLM_DSPARK_DYNAMIC_CONFIDENCE_TRACE_PATH"

__all__ = [
    "CONFIDENCE_TRACE_PATH_ENV",
    "DYNAMIC_CONFIDENCE_TRACE_PATH_ENV",
    "ConfidenceTraceRing",
    "confidence_trace_path_from_env",
    "dynamic_confidence_trace_path_from_env",
]


def confidence_trace_path_from_env() -> str | None:
    """Return the configured trace stem, or ``None`` when collection is off."""
    value = os.environ.get(CONFIDENCE_TRACE_PATH_ENV, "").strip()
    return value or None


def dynamic_confidence_trace_path_from_env() -> str | None:
    """Return the opt-in dynamic-policy trace stem, if configured."""
    value = os.environ.get(DYNAMIC_CONFIDENCE_TRACE_PATH_ENV, "").strip()
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
        dynamic_diagnostic: bool = False,
        device: torch.device | str = "cuda",
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
        self.dynamic_diagnostic = dynamic_diagnostic
        self._flushed = False

        self._slot_logits = torch.zeros(
            (num_slots, max_draft_len), dtype=torch.float32, device=device
        )
        self._slot_valid = torch.zeros(num_slots, dtype=torch.bool, device=device)
        self._slot_graph_batch_size = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self._slot_step_id = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self._logits = torch.zeros((capacity, max_draft_len), dtype=torch.float32, device=device)
        self._prefix_mask = torch.zeros_like(self._logits)
        self._valid = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._graph_batch_size = torch.zeros(capacity, dtype=torch.int32, device=device)
        self._step_id = torch.zeros(capacity, dtype=torch.int64, device=device)
        self._counter = torch.zeros((), dtype=torch.int64, device=device)
        self._event_counter = torch.zeros((), dtype=torch.int64, device=device)
        if dynamic_diagnostic:
            self._slot_request_id = torch.full(
                (num_slots,), -1, dtype=torch.int64, device=device
            )
            self._slot_generation_progress = torch.full_like(self._slot_request_id, -1)
            self._slot_real_mask = torch.zeros(num_slots, dtype=torch.bool, device=device)
            self._slot_selected_budget = torch.zeros(
                num_slots, dtype=torch.int32, device=device
            )
            self._slot_compact_budget = torch.zeros_like(self._slot_selected_budget)
            self._slot_full_budget = torch.zeros_like(self._slot_selected_budget)
            self._slot_compact_goodput = torch.zeros(
                num_slots, dtype=torch.float32, device=device
            )
            self._slot_full_goodput = torch.zeros_like(self._slot_compact_goodput)

            self._slot_id = torch.zeros(capacity, dtype=torch.int64, device=device)
            self._request_id = torch.full_like(self._slot_id, -1)
            self._generation_progress = torch.full_like(self._slot_id, -1)
            self._real_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
            self._accepted_token_count = torch.zeros(
                capacity, dtype=torch.int32, device=device
            )
            self._accepted_draft_count = torch.zeros_like(self._accepted_token_count)
            self._selected_budget = torch.zeros_like(self._accepted_token_count)
            self._compact_budget = torch.zeros_like(self._accepted_token_count)
            self._full_budget = torch.zeros_like(self._accepted_token_count)
            self._compact_goodput = torch.zeros(
                capacity, dtype=torch.float32, device=device
            )
            self._full_goodput = torch.zeros_like(self._compact_goodput)
            self._compact_full_margin = torch.zeros_like(self._compact_goodput)
        atexit.register(self.flush)
        logger.warning(
            f"DSpark confidence tracing enabled: capacity={capacity} rows, "
            f"output={self.path}. Collection is diagnostic and adds device memory."
        )

    def invalidate_slot(self, slot: int) -> None:
        """Prevent a recycled request slot from pairing with stale logits."""
        self._slot_valid[slot] = False
        if self.dynamic_diagnostic:
            self._slot_request_id[slot] = -1
            self._slot_generation_progress[slot] = -1
            self._slot_real_mask[slot] = False

    def bind_slot_identity(self, slot: int, request_id: int) -> None:
        """Bind a stable request identity to a lifecycle-managed worker slot."""
        if self.dynamic_diagnostic:
            self._slot_request_id[slot] = int(request_id)

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
        self._graph_batch_size.index_copy_(0, row_indices, previous_graph_batch_size)
        self._step_id.index_copy_(0, row_indices, previous_step_id)
        if self.dynamic_diagnostic:
            # Padding rows have no stable request slot, but retaining them is
            # essential for reconstructing the exact ADP real-mask roster.
            previous_valid = torch.where(
                real_rows,
                previous_valid,
                self._event_counter > 0,
            )
            previous_real_mask = self._slot_real_mask.index_select(0, slots)
            accepted_tokens = torch.where(
                previous_real_mask,
                num_accepted_tokens.to(torch.int32),
                torch.zeros((), dtype=torch.int32, device=slots.device),
            )
            self._slot_id.index_copy_(0, row_indices, slots)
            self._request_id.index_copy_(
                0, row_indices, self._slot_request_id.index_select(0, slots)
            )
            self._generation_progress.index_copy_(
                0, row_indices, self._slot_generation_progress.index_select(0, slots)
            )
            self._real_mask.index_copy_(0, row_indices, previous_real_mask)
            self._accepted_token_count.index_copy_(0, row_indices, accepted_tokens)
            self._accepted_draft_count.index_copy_(
                0, row_indices, (accepted_tokens - 1).clamp(min=0, max=self.max_draft_len)
            )
            self._selected_budget.index_copy_(
                0, row_indices, self._slot_selected_budget.index_select(0, slots)
            )
            self._compact_budget.index_copy_(
                0, row_indices, self._slot_compact_budget.index_select(0, slots)
            )
            self._full_budget.index_copy_(
                0, row_indices, self._slot_full_budget.index_select(0, slots)
            )
            compact_goodput = self._slot_compact_goodput.index_select(0, slots)
            full_goodput = self._slot_full_goodput.index_select(0, slots)
            self._compact_goodput.index_copy_(0, row_indices, compact_goodput)
            self._full_goodput.index_copy_(0, row_indices, full_goodput)
            self._compact_full_margin.index_copy_(
                0, row_indices, compact_goodput - full_goodput
            )
        self._valid.index_copy_(0, row_indices, previous_valid)
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

    def publish_dynamic_plan(
        self,
        *,
        slots: torch.Tensor,
        real_request_mask: torch.Tensor,
        generation_progress: torch.Tensor,
        selected_budget: torch.Tensor,
        compact_budget: torch.Tensor,
        full_budget: torch.Tensor,
        compact_goodput: torch.Tensor,
        full_goodput: torch.Tensor,
    ) -> None:
        """Attach the current dynamic policy decision to its slot-owned logits."""
        if not self.dynamic_diagnostic:
            raise RuntimeError("dynamic-plan publication requires dynamic diagnostics")
        num_requests = slots.shape[0]
        vectors = (
            real_request_mask,
            generation_progress,
            compact_goodput,
            full_goodput,
        )
        if any(value.shape != (num_requests,) for value in vectors):
            raise ValueError("dynamic diagnostic row tensors must have shape [G]")
        if selected_budget.ndim or compact_budget.ndim or full_budget.ndim:
            raise ValueError("dynamic diagnostic budgets must be scalar tensors")
        slots = slots.to(torch.long)
        real_request_mask = real_request_mask.to(torch.bool)
        self._slot_generation_progress.index_copy_(
            0, slots, torch.where(real_request_mask, generation_progress, -1)
        )
        self._slot_real_mask.index_copy_(0, slots, real_request_mask)
        self._slot_selected_budget.index_copy_(
            0, slots, selected_budget.to(torch.int32).expand(num_requests)
        )
        self._slot_compact_budget.index_copy_(
            0, slots, compact_budget.to(torch.int32).expand(num_requests)
        )
        self._slot_full_budget.index_copy_(
            0, slots, full_budget.to(torch.int32).expand(num_requests)
        )
        self._slot_compact_goodput.index_copy_(
            0, slots, torch.where(real_request_mask, compact_goodput.float(), 0.0)
        )
        self._slot_full_goodput.index_copy_(
            0, slots, torch.where(real_request_mask, full_goodput.float(), 0.0)
        )

    @torch.inference_mode()
    def reset(self) -> None:
        """Discard warmup/capture state while preserving captured buffer addresses."""
        if self._flushed:
            raise RuntimeError("cannot reset a flushed DSpark confidence trace ring")
        self._slot_valid.zero_()
        self._valid.zero_()
        self._counter.zero_()
        self._event_counter.zero_()
        if self.dynamic_diagnostic:
            self._slot_generation_progress.fill_(-1)
            self._slot_real_mask.zero_()
        logger.warning("DSpark confidence trace: discarded warmup and CUDA-capture rows")

    def flush(self) -> None:
        """Write valid paired rows to a CPU shard once."""
        if self._flushed:
            return
        self._flushed = True
        try:
            if self._logits.is_cuda:
                torch.cuda.synchronize()
            counter = int(self._counter.item())
            written = min(counter, self.capacity)
            if written == 0:
                logger.warning("DSpark confidence trace ring is empty; no shard written")
                return
            if counter <= self.capacity:
                order = torch.arange(written, device=self._logits.device)
            else:
                start = counter % self.capacity
                order = torch.cat(
                    (
                        torch.arange(start, self.capacity, device=self._logits.device),
                        torch.arange(0, start, device=self._logits.device),
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
            if self.dynamic_diagnostic:
                payload.update(
                    {
                        "diagnostic_schema": "dspark_dynamic_policy_trace_v1",
                        "slot_id": self._slot_id.index_select(0, order).cpu(),
                        "request_id": self._request_id.index_select(0, order).cpu(),
                        "generation_progress": self._generation_progress.index_select(
                            0, order
                        ).cpu(),
                        "real_mask": self._real_mask.index_select(0, order).cpu(),
                        "accepted_token_count": self._accepted_token_count.index_select(
                            0, order
                        ).cpu(),
                        "accepted_draft_count": self._accepted_draft_count.index_select(
                            0, order
                        ).cpu(),
                        "selected_budget": self._selected_budget.index_select(0, order).cpu(),
                        "compact_budget": self._compact_budget.index_select(0, order).cpu(),
                        "full_budget": self._full_budget.index_select(0, order).cpu(),
                        "predicted_compact_goodput": self._compact_goodput.index_select(
                            0, order
                        ).cpu(),
                        "predicted_full_goodput": self._full_goodput.index_select(
                            0, order
                        ).cpu(),
                        "predicted_compact_minus_full_margin": (
                            self._compact_full_margin.index_select(0, order).cpu()
                        ),
                        "counterfactual_full_k_acceptance_available": False,
                        "counterfactual_full_k_acceptance": None,
                        "acceptance_semantics": (
                            "accepted counts are observed under selected_budget; "
                            "counterfactual full-K acceptance is unavailable"
                        ),
                    }
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.path)
            logger.warning(
                f"DSpark confidence trace: wrote {payload['rows_written']} paired "
                f"rows to {self.path}"
            )
        except (OSError, RuntimeError) as error:
            logger.error(f"Failed to flush DSpark confidence trace {self.path}: {error}")
