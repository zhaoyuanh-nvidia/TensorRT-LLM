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
"""GPU-only planning for fixed- and dynamic-budget DSpark verification."""

import json
from pathlib import Path
from typing import Callable, NamedTuple

import torch

__all__ = [
    "DynamicBudgetPlan",
    "FixedBudgetPlan",
    "apply_sts",
    "load_sts_temperatures",
    "plan_dynamic_verifier_budget",
    "plan_fixed_verifier_budget",
    "verify_packed_greedy",
]


class DynamicBudgetPlan(NamedTuple):
    """Fixed-shape result of selecting one captured verifier-token tier."""

    retained_lens: torch.Tensor
    verifier_token_budget: torch.Tensor


class FixedBudgetPlan(NamedTuple):
    """Device tensors describing a fixed-budget verifier workload.

    ``packed_to_dense`` maps packed verifier rows to flattened dense
    ``[num_requests, max_draft_len + 1]`` rows. Local dense position zero is
    the anchor token and positions one through K are draft tokens.
    ``dense_to_packed`` contains the inverse map and uses -1 for omitted
    drafts.
    """

    retained_lens: torch.Tensor
    query_lens: torch.Tensor
    cu_query_lens: torch.Tensor
    packed_to_dense: torch.Tensor
    dense_to_packed: torch.Tensor
    packed_request_ids: torch.Tensor
    packed_local_positions: torch.Tensor


def load_sts_temperatures(path: str | Path, max_draft_len: int) -> torch.Tensor:
    """Load a per-position sequential-temperature-scaling vector.

    TensorRT-LLM emits ``sts_temperatures`` while SGLang emits
    ``temperatures``.  Accept both spellings so a calibration fitted from the
    same checkpoint/head/block geometry is portable between the runtimes.
    """
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    artifact_max_draft_len = payload.get("max_draft_len")
    if artifact_max_draft_len is not None and int(artifact_max_draft_len) != max_draft_len:
        raise ValueError(
            f"STS artifact max_draft_len={artifact_max_draft_len} does not match "
            f"runtime max_draft_len={max_draft_len}"
        )
    if "sts_temperatures" in payload and "temperatures" in payload:
        sts_values = torch.as_tensor(payload["sts_temperatures"], dtype=torch.float32)
        compatibility_values = torch.as_tensor(payload["temperatures"], dtype=torch.float32)
        if not torch.equal(sts_values, compatibility_values):
            raise ValueError(
                "STS artifact contains inconsistent 'sts_temperatures' and 'temperatures' vectors"
            )
    values = payload.get("sts_temperatures", payload.get("temperatures"))
    if values is None:
        raise ValueError(f"{path} has neither 'sts_temperatures' nor 'temperatures'")
    temperatures = torch.tensor(values, dtype=torch.float32)
    if temperatures.ndim != 1 or temperatures.numel() != max_draft_len:
        raise ValueError(
            f"STS temperature vector must have length {max_draft_len}; "
            f"got shape {tuple(temperatures.shape)}"
        )
    if not torch.all(torch.isfinite(temperatures)) or not torch.all(temperatures > 0):
        raise ValueError("STS temperatures must be positive and finite")
    return temperatures


def apply_sts(
    confidence_logits: torch.Tensor,
    temperatures: torch.Tensor | None,
) -> torch.Tensor:
    """Convert raw confidence logits to calibrated conditional probabilities.

    A NaN confidence is conservatively ranked as zero survival probability.
    """
    logits = torch.nan_to_num(confidence_logits.float(), nan=float("-inf"))
    if temperatures is None:
        return torch.sigmoid(logits)
    if temperatures.ndim != 1 or temperatures.numel() != logits.shape[-1]:
        raise ValueError("temperatures must be a vector matching the confidence-logit width")
    return torch.sigmoid(logits / temperatures.to(device=logits.device, dtype=torch.float32))


def plan_dynamic_verifier_budget(
    confidence_logits: torch.Tensor,
    candidate_budgets: torch.Tensor,
    candidate_step_times_ms: torch.Tensor,
    minimum_predicted_gain: float = 0.01,
    temperatures: torch.Tensor | None = None,
    candidate_yield_reducer: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> DynamicBudgetPlan:
    """Select and allocate one captured verifier-token budget on device.

    The small candidate ladder and its measured ``T(G,V)`` values are static
    tensors.  Confidence determines the expected accepted-token yield of each
    tier, and the selected tier maximizes predicted yield divided by measured
    step time.  The full-K tier must be the final (largest) candidate and is
    retained unless a compact tier clears ``minimum_predicted_gain``.

    All output shapes are independent of the selected tier.  This lets the
    planner run inside the drafter CUDA graph; the existing asynchronous
    ``next_draft_lens`` copy carries both the per-request allocation and its
    total to the host, where the following target step selects the matching
    pre-captured ``(G,V)`` graph.  No confidence snapshot or additional D2H
    transfer is introduced.

    ``candidate_yield_reducer`` optionally aggregates the fixed-size yield
    vector before scoring. Attention-DP uses it to sum yield across ranks so
    every rank selects one globally optimal graph tier while retaining a
    rank-local, exact-size ragged allocation. A one-tier ladder skips the
    reducer because the selected graph tier is configuration-invariant; this
    avoids a redundant captured collective without changing the allocation.
    """
    if confidence_logits.ndim != 2:
        raise ValueError("confidence_logits must have shape [num_requests, max_draft_len]")
    num_requests, max_draft_len = confidence_logits.shape
    if num_requests == 0 or max_draft_len == 0:
        raise ValueError("confidence_logits must have non-zero dimensions")
    if candidate_budgets.ndim != 1 or candidate_step_times_ms.ndim != 1:
        raise ValueError("candidate budgets and step times must be vectors")
    if candidate_budgets.numel() == 0 or (
        candidate_budgets.numel() != candidate_step_times_ms.numel()
    ):
        raise ValueError("candidate budgets and step times must have the same non-zero length")
    if minimum_predicted_gain < 0.0:
        raise ValueError("minimum_predicted_gain must be non-negative")

    dense_capacity = num_requests * (max_draft_len + 1)
    if candidate_budgets.device != confidence_logits.device:
        raise ValueError("candidate budgets must be on the confidence-logit device")
    if candidate_step_times_ms.device != confidence_logits.device:
        raise ValueError("candidate step times must be on the confidence-logit device")
    # Candidate values are configuration constants validated before they are
    # copied to the GPU.  Shape/range checks here intentionally avoid reading
    # their contents back to the host during capture.
    if candidate_budgets.numel() > dense_capacity:
        raise ValueError("candidate ladder cannot exceed dense verifier capacity")

    conditional = apply_sts(confidence_logits, temperatures)
    survival = torch.cumprod(conditional, dim=1)
    position_major = survival.transpose(0, 1).contiguous().view(-1)
    selection_order = torch.argsort(position_major, descending=True, stable=True)
    ordered_survival = position_major[selection_order]
    cumulative_yield = torch.cat(
        (
            torch.zeros(1, dtype=torch.float32, device=confidence_logits.device),
            torch.cumsum(ordered_survival, dim=0, dtype=torch.float32),
        )
    )

    retained_candidates = candidate_budgets.to(torch.long) - num_requests
    expected_yield = float(num_requests) + torch.gather(cumulative_yield, 0, retained_candidates)
    # Cross-rank yield is only needed to choose among multiple graph tiers.
    # A one-tier ladder has a configuration-invariant selection, so bypassing
    # the reducer avoids a redundant captured collective without changing the
    # selected budget or its rank-local ragged allocation.
    if candidate_yield_reducer is not None and candidate_budgets.numel() > 1:
        expected_yield = candidate_yield_reducer(expected_yield.contiguous())
        if expected_yield.shape != candidate_budgets.shape:
            raise ValueError("candidate_yield_reducer must preserve the candidate shape")
        if expected_yield.device != confidence_logits.device:
            raise ValueError("candidate_yield_reducer must preserve the confidence-logit device")
    scores = expected_yield.to(torch.float32) / candidate_step_times_ms.to(torch.float32)

    # Prefer the larger tier on an exact score tie, matching the offline
    # planner and making the conservative choice deterministic.
    reverse_best = torch.argmax(torch.flip(scores, dims=(0,)))
    best_index = scores.numel() - 1 - reverse_best
    full_index = torch.full_like(best_index, scores.numel() - 1)
    best_score = torch.gather(scores, 0, best_index.reshape(1)).squeeze(0)
    full_score = scores[-1]
    clears_guard = best_score >= full_score * (1.0 + float(minimum_predicted_gain))
    selected_index = torch.where(clears_guard, best_index, full_index)
    selected_retained = torch.gather(retained_candidates, 0, selected_index.reshape(1)).squeeze(0)

    # Convert the global stable rank to a prefix-closed per-request length.
    # The comparison against a device scalar has a fixed output shape even
    # though the selected V changes between graph replays.
    selection_ranks = torch.empty_like(selection_order)
    selection_ranks.scatter_(
        0,
        selection_order,
        torch.arange(selection_order.numel(), device=confidence_logits.device),
    )
    selected_position_major = selection_ranks < selected_retained
    selected_drafts = selected_position_major.view(max_draft_len, num_requests).transpose(0, 1)
    retained_lens = selected_drafts.sum(dim=1, dtype=torch.int32)
    return DynamicBudgetPlan(
        retained_lens=retained_lens,
        verifier_token_budget=torch.gather(candidate_budgets, 0, selected_index.reshape(1)).squeeze(
            0
        ),
    )


def plan_fixed_verifier_budget(
    confidence_logits: torch.Tensor,
    verifier_token_budget: int,
    temperatures: torch.Tensor | None = None,
) -> FixedBudgetPlan:
    """Allocate an exact, fixed verifier budget using confidence logits.

    The verifier budget includes one mandatory anchor token per request. The
    remaining budget is assigned to draft-token prefixes by ranking their
    estimated survival probabilities. For draft position ``j``, the survival
    probability is the cumulative product of conditional confidence through
    ``j``. These scores are non-increasing within a request, so global
    selection is prefix-closed.

    Equal scores prefer an earlier draft position, then an earlier request.
    The implementation is tensor-only after static shape/budget validation and
    is suitable for CUDA graph capture. It performs no device-to-host transfer
    and creates no data-dependent-size tensor.

    Args:
        confidence_logits: Conditional-acceptance logits with shape
            ``[num_requests, max_draft_len]``. The tensor may reside on CPU or
            GPU.
        temperatures: Optional per-position STS vector. ``None`` preserves the
            identity-temperature behavior.
        verifier_token_budget: Exact number of packed verifier tokens,
            including one anchor for every request.

    Returns:
        A :class:`FixedBudgetPlan` whose packed maps have exactly
        ``verifier_token_budget`` entries and whose dense inverse map has
        ``num_requests * (max_draft_len + 1)`` entries.

    Raises:
        ValueError: If the logits are not a non-empty matrix or if the budget
            cannot provide one anchor per request or exceeds dense K+1
            capacity.
    """
    if confidence_logits.ndim != 2:
        raise ValueError("confidence_logits must have shape [num_requests, max_draft_len]")

    num_requests, max_draft_len = confidence_logits.shape
    if num_requests == 0 or max_draft_len == 0:
        raise ValueError("confidence_logits must have non-zero dimensions")

    dense_width = max_draft_len + 1
    dense_capacity = num_requests * dense_width
    if not num_requests <= verifier_token_budget <= dense_capacity:
        raise ValueError(
            "verifier_token_budget must be between num_requests and "
            "num_requests * (max_draft_len + 1), inclusive"
        )

    num_retained = verifier_token_budget - num_requests

    # A draft at position j is useful only if every conditional prediction up
    # to j is accepted. Compute in FP32 so the ranking remains useful when the
    # head or model executes in a lower precision.
    conditional_probabilities = apply_sts(confidence_logits, temperatures)
    prefix_scores = torch.cumprod(conditional_probabilities, dim=1)

    # Stable sorting a position-major view implements the deterministic tie
    # order (position, request). It also preserves prefix closure when scores
    # saturate or underflow to equal values.
    position_major_scores = prefix_scores.transpose(0, 1).contiguous().view(-1)
    selection_order = torch.argsort(position_major_scores, descending=True, stable=True)
    selected_position_major = torch.zeros_like(position_major_scores, dtype=torch.bool)
    selected_position_major.scatter_(0, selection_order[:num_retained], True)
    selected_drafts = selected_position_major.view(max_draft_len, num_requests).transpose(0, 1)

    retained_lens = selected_drafts.sum(dim=1, dtype=torch.int32)
    query_lens = retained_lens + 1
    cu_query_lens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=confidence_logits.device),
            torch.cumsum(query_lens, dim=0, dtype=torch.int32),
        )
    )

    local_positions = torch.arange(dense_width, dtype=torch.int32, device=confidence_logits.device)
    dense_keep_mask = local_positions.unsqueeze(0) < query_lens.unsqueeze(1)
    dense_keep_mask = dense_keep_mask.reshape(-1)

    # Selected dense rows receive keys in [0, dense_capacity); omitted rows
    # receive larger keys. Sorting therefore packs selected rows in the dense
    # request-major order without a data-dependent nonzero operation.
    dense_indices = torch.arange(dense_capacity, dtype=torch.int64, device=confidence_logits.device)
    pack_sort_keys = torch.where(dense_keep_mask, dense_indices, dense_indices + dense_capacity)
    packed_to_dense = torch.argsort(pack_sort_keys)[:verifier_token_budget]

    dense_to_packed = torch.full(
        (dense_capacity,), -1, dtype=torch.int64, device=confidence_logits.device
    )
    dense_to_packed.scatter_(
        0,
        packed_to_dense,
        torch.arange(
            verifier_token_budget,
            dtype=torch.int64,
            device=confidence_logits.device,
        ),
    )

    packed_request_ids = torch.div(packed_to_dense, dense_width, rounding_mode="floor")
    packed_local_positions = torch.remainder(packed_to_dense, dense_width)

    return FixedBudgetPlan(
        retained_lens=retained_lens,
        query_lens=query_lens,
        cu_query_lens=cu_query_lens,
        packed_to_dense=packed_to_dense,
        dense_to_packed=dense_to_packed,
        packed_request_ids=packed_request_ids,
        packed_local_positions=packed_local_positions,
    )


def verify_packed_greedy(
    target_tokens: torch.Tensor,
    packed_draft_tokens: torch.Tensor,
    draft_lens: torch.Tensor,
    max_draft_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify packed target tokens against packed per-request draft prefixes.

    ``target_tokens`` contains each request's ``d_i`` verifier predictions
    followed by its golden token, while ``packed_draft_tokens`` contains only
    the ``d_i`` draft tokens. Both inputs are request-major and their sizes are
    fixed by the selected verifier budget. Returned accepted-token storage
    keeps the physical ``K+1`` width used by the shared speculative sampler.
    """
    if draft_lens.ndim != 1:
        raise ValueError("draft_lens must have shape [num_requests]")
    if max_draft_len < 1:
        raise ValueError("max_draft_len must be greater than zero")

    batch_size = draft_lens.shape[0]
    device = target_tokens.device
    draft_lens = draft_lens.to(device=device, dtype=torch.long)
    query_lens = draft_lens + 1
    cu_query_lens = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(query_lens, dim=0),
        )
    )
    cu_draft_lens = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(draft_lens, dim=0),
        )
    )

    local_query_positions = torch.arange(max_draft_len + 1, device=device)
    query_indices = cu_query_lens[:-1].unsqueeze(1) + local_query_positions
    query_indices = query_indices.clamp(max=target_tokens.shape[0] - 1)
    dense_target_tokens = target_tokens[query_indices].to(torch.int32)
    valid_queries = local_query_positions.unsqueeze(0) < query_lens.unsqueeze(1)
    accepted_tokens = torch.where(
        valid_queries,
        dense_target_tokens,
        torch.zeros((), dtype=torch.int32, device=device),
    )

    local_draft_positions = torch.arange(max_draft_len, device=device)
    valid_drafts = local_draft_positions.unsqueeze(0) < draft_lens.unsqueeze(1)
    if packed_draft_tokens.numel() == 0:
        dense_draft_tokens = torch.zeros(
            (batch_size, max_draft_len), dtype=torch.int32, device=device
        )
    else:
        draft_indices = (cu_draft_lens[:-1].unsqueeze(1) + local_draft_positions).clamp(
            max=packed_draft_tokens.shape[0] - 1
        )
        dense_draft_tokens = packed_draft_tokens[draft_indices]

    matches = (dense_draft_tokens == dense_target_tokens[:, :max_draft_len]) & valid_drafts
    num_accepted_tokens = 1 + torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1)
    return accepted_tokens, num_accepted_tokens
