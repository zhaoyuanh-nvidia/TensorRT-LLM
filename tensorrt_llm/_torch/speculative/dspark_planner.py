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
"""Cost-aware offline planning for fixed-budget DSpark verification.

The runtime deliberately captures one exact verifier-token budget per CUDA
graph batch size.  This module turns a sparse, measured step-cost surface and
confidence-head traces into that schedule.  Planning stays offline so serving
does not add a device-to-host synchronization or a second CUDA-graph axis.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "SpsCostTable",
    "derive_fixed_verifier_budget_candidates",
    "load_sps_cost_table",
    "select_fixed_verifier_budget",
    "select_fixed_verifier_budget_from_traces",
    "validate_sps_cost_table_payload",
]


@dataclass(frozen=True)
class SpsCostTable:
    """Measured decode step time as a function of submitted verifier tokens.

    ``step_time_ms`` is the token-dependent term at each ``token_counts``
    breakpoint. ``fixed_overhead_ms`` and the optional batch-size axis carry
    costs that verification trimming cannot remove.  All three terms matter
    because the planner maximizes a ratio, not a difference.
    """

    token_counts: Sequence[int]
    step_time_ms: Sequence[float]
    fixed_overhead_ms: float = 0.0
    batch_sizes: Sequence[int] = field(default_factory=tuple)
    batch_overhead_ms: Sequence[float] = field(default_factory=tuple)
    minimum_predicted_gain: float = 0.01

    def __post_init__(self) -> None:
        if len(self.token_counts) != len(self.step_time_ms):
            raise ValueError("token_counts and step_time_ms must have the same length")
        if not self.token_counts:
            raise ValueError("SpsCostTable requires at least one measured point")
        if any(right <= left for left, right in zip(self.token_counts, self.token_counts[1:])):
            raise ValueError("token_counts must be strictly increasing")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.step_time_ms):
            raise ValueError("step_time_ms entries must be positive and finite")
        if not math.isfinite(self.fixed_overhead_ms) or self.fixed_overhead_ms < 0.0:
            raise ValueError("fixed_overhead_ms must be non-negative and finite")
        if len(self.batch_sizes) != len(self.batch_overhead_ms):
            raise ValueError("batch_sizes and batch_overhead_ms must have the same length")
        if any(right <= left for left, right in zip(self.batch_sizes, self.batch_sizes[1:])):
            raise ValueError("batch_sizes must be strictly increasing")
        if any(not math.isfinite(value) or value < 0.0 for value in self.batch_overhead_ms):
            raise ValueError("batch_overhead_ms entries must be non-negative and finite")
        if not math.isfinite(self.minimum_predicted_gain) or self.minimum_predicted_gain < 0.0:
            raise ValueError("minimum_predicted_gain must be non-negative and finite")

    @property
    def is_flat(self) -> bool:
        """Whether the measured token-dependent term has no variation."""
        return len(set(float(value) for value in self.step_time_ms)) <= 1

    def batch_overhead(self, num_requests: int) -> float:
        """Return the interpolated non-trimmable batch-size overhead."""
        if not self.batch_sizes:
            return 0.0
        return float(
            np.interp(
                float(num_requests),
                np.asarray(self.batch_sizes, dtype=np.float64),
                np.asarray(self.batch_overhead_ms, dtype=np.float64),
            )
        )

    def step_times(self, num_tokens: np.ndarray, num_requests: int) -> np.ndarray:
        """Return whole-step milliseconds for one or more token counts."""
        token_term = np.interp(
            np.asarray(num_tokens, dtype=np.float64),
            np.asarray(self.token_counts, dtype=np.float64),
            np.asarray(self.step_time_ms, dtype=np.float64),
        )
        return token_term + self.fixed_overhead_ms + self.batch_overhead(num_requests)

    def step_time(self, num_tokens: int, num_requests: int) -> float:
        """Return whole-step milliseconds for one token count."""
        return float(self.step_times(np.asarray([num_tokens]), num_requests)[0])


def load_sps_cost_table(path: str | Path) -> tuple[SpsCostTable, dict[str, object]]:
    """Load the profiler JSON schema shared with PR #17056 and SGLang.

    The returned payload is retained so callers can validate its engine
    fingerprint before using the numbers.
    """
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    table = SpsCostTable(
        token_counts=tuple(int(value) for value in payload["token_counts"]),
        step_time_ms=tuple(float(value) for value in payload["step_time_ms"]),
        fixed_overhead_ms=float(payload.get("fixed_overhead_ms", 0.0)),
        batch_sizes=tuple(int(value) for value in payload.get("batch_sizes", ())),
        batch_overhead_ms=tuple(float(value) for value in payload.get("batch_overhead_ms", ())),
        minimum_predicted_gain=float(payload.get("minimum_predicted_gain", 0.01)),
    )
    return table, payload


def validate_sps_cost_table_payload(
    payload: dict[str, object],
    *,
    verifier_token_budget_tiers: dict[int, Sequence[int]],
    max_draft_len: int,
) -> dict[str, object]:
    """Fail closed when a dynamic cost artifact does not match its runtime.

    Dynamic serving must only replay tiers measured for the same local graph
    request width and K.  A one-dimensional SPS artifact represents one exact
    graph batch size; accepting an unmeasured tier would silently interpolate
    a different CUDA-graph shape and defeat the measured ``T(G, V)`` contract.
    """
    if not isinstance(payload, dict):
        raise TypeError("SPS cost artifact must contain a JSON object")
    fingerprint = payload.get("engine_fingerprint")
    if not isinstance(fingerprint, dict):
        raise TypeError("SPS cost artifact requires an engine_fingerprint object")

    required_fingerprint_keys = {
        "gpu",
        "gpu_count",
        "max_draft_len",
        "rank_local_graph_batch_size",
        "runtime_snapshot",
        "source_head",
        "topology",
    }
    missing_fingerprint_keys = sorted(required_fingerprint_keys - fingerprint.keys())
    if missing_fingerprint_keys:
        raise ValueError(
            "SPS engine_fingerprint is missing required keys: "
            + ", ".join(missing_fingerprint_keys)
        )

    normalized_tiers = {
        int(graph_batch_size): tuple(int(value) for value in values)
        for graph_batch_size, values in verifier_token_budget_tiers.items()
    }
    if len(normalized_tiers) != 1:
        raise ValueError(
            "A one-dimensional SPS cost artifact must configure exactly one "
            "rank-local graph batch size"
        )
    graph_batch_size, candidate_budgets = next(iter(normalized_tiers.items()))
    if int(fingerprint["rank_local_graph_batch_size"]) != graph_batch_size:
        raise ValueError(
            "SPS engine_fingerprint rank_local_graph_batch_size does not match "
            f"configured G={graph_batch_size}"
        )
    if int(fingerprint["max_draft_len"]) != max_draft_len:
        raise ValueError(
            f"SPS engine_fingerprint max_draft_len does not match runtime K={max_draft_len}"
        )

    measured_budgets = {int(value) for value in payload.get("token_counts", ())}
    unmeasured_budgets = sorted(set(candidate_budgets) - measured_budgets)
    if unmeasured_budgets:
        raise ValueError(
            "Dynamic verifier tiers must be directly measured SPS token_counts; "
            f"unmeasured tiers: {unmeasured_budgets}"
        )
    full_budget = graph_batch_size * (max_draft_len + 1)
    if full_budget not in candidate_budgets:
        raise ValueError(f"Dynamic verifier tiers must include full-K fallback V={full_budget}")

    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("SPS cost artifact requires non-empty measurement provenance")
    provenance_budgets = {
        int(item["rank_local_verifier_budget"])
        for item in measurements
        if isinstance(item, dict) and "rank_local_verifier_budget" in item
    }
    missing_provenance = sorted(set(candidate_budgets) - provenance_budgets)
    if missing_provenance:
        raise ValueError(
            f"SPS measurement provenance does not cover configured tiers: {missing_provenance}"
        )

    gpu_count = int(fingerprint["gpu_count"])
    if gpu_count < 1:
        raise ValueError("SPS engine_fingerprint gpu_count must be positive")
    global_graph_batch_size = fingerprint.get("global_graph_batch_size")
    if (
        global_graph_batch_size is not None
        and int(global_graph_batch_size) != graph_batch_size * gpu_count
    ):
        raise ValueError(
            "SPS engine_fingerprint global_graph_batch_size is inconsistent "
            "with rank-local G and gpu_count"
        )
    return fingerprint


def derive_fixed_verifier_budget_candidates(
    *,
    cost_table: SpsCostTable,
    num_requests: int,
    max_draft_len: int,
    max_candidates: int = 4,
) -> tuple[int, ...]:
    """Derive a small exact-V ladder from a sparse measured cost curve.

    Candidate graphs are placed at measured breakpoints inside the reachable
    range.  The full K budget is always retained as the safe static fallback.
    The lower bound submits at least one draft plus one anchor per request;
    individual rows can still receive zero drafts after global allocation.
    When the measured curve has more breakpoints than the graph-memory budget,
    keep the interior points with the largest adjacent cost changes.
    """
    if num_requests < 1:
        raise ValueError("num_requests must be positive")
    if max_draft_len < 1:
        raise ValueError("max_draft_len must be positive")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    minimum = num_requests * 2
    full = num_requests * (max_draft_len + 1)
    reachable = {
        int(tokens) for tokens in cost_table.token_counts if minimum <= int(tokens) <= full
    }
    reachable.add(full)
    ordered = sorted(reachable)
    if len(ordered) <= max_candidates:
        return tuple(ordered)
    if max_candidates == 1:
        return (full,)

    interior = [value for value in ordered if value != full]
    ranked: list[tuple[float, int]] = []
    for value in interior:
        index = list(cost_table.token_counts).index(value)
        previous = max(index - 1, 0)
        delta = abs(
            float(cost_table.step_time_ms[index]) - float(cost_table.step_time_ms[previous])
        )
        ranked.append((delta, value))
    selected = [value for _, value in sorted(ranked, reverse=True)[: max_candidates - 1]]
    return tuple(sorted({*selected, full}))


def _expected_yield_curve(survival: np.ndarray, num_requests: int) -> np.ndarray:
    """Expected accepted-token yield for every retained-draft count."""
    candidates = np.sort(np.asarray(survival, dtype=np.float64).reshape(-1))[::-1]
    return float(num_requests) + np.concatenate(([0.0], np.cumsum(candidates)))


def select_fixed_verifier_budget(
    *,
    survival: np.ndarray,
    candidate_budgets: Sequence[int],
    cost_table: SpsCostTable,
) -> int:
    """Select the exact verifier budget with the best predicted goodput.

    ``survival[r, j]`` is the probability that request ``r`` accepts every
    draft through position ``j``.  For budget V, V-G draft positions are
    granted globally in descending survival order, matching the runtime's
    prefix-closed allocator.  A trimmed candidate must beat full K by the
    table's ``minimum_predicted_gain``; otherwise full K wins.  This guard
    turns profiler/calibration noise into a no-op instead of a regression.
    """
    values = np.asarray(survival, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"survival must have shape [G, K], got {values.shape}")
    num_requests, max_draft_len = values.shape
    if num_requests == 0 or max_draft_len == 0:
        raise ValueError("survival must have non-zero dimensions")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("survival entries must be finite probabilities")

    minimum = num_requests
    full = num_requests * (max_draft_len + 1)
    budgets = sorted({int(value) for value in candidate_budgets})
    if full not in budgets:
        budgets.append(full)
    if any(value < minimum or value > full for value in budgets):
        raise ValueError(f"candidate budgets must lie in [{minimum}, {full}]")
    if cost_table.is_flat:
        return full

    expected_yield = _expected_yield_curve(values, num_requests)
    scores: dict[int, float] = {}
    for budget in budgets:
        retained = budget - num_requests
        scores[budget] = float(expected_yield[retained]) / cost_table.step_time(
            budget, num_requests
        )

    full_score = scores[full]
    best = max(budgets, key=lambda budget: (scores[budget], budget))
    if best != full and scores[best] < full_score * (1.0 + cost_table.minimum_predicted_gain):
        return full
    return best


def select_fixed_verifier_budget_from_traces(
    *,
    survival_steps: np.ndarray,
    candidate_budgets: Sequence[int],
    cost_table: SpsCostTable,
    prefix_mask_steps: np.ndarray | None = None,
) -> tuple[int, dict[int, float]]:
    """Choose one exact budget for a batch size from many full-K traces.

    Args:
        survival_steps: Calibrated prefix-survival probabilities with shape
            ``[num_steps, G, K]`` collected while verifying the full block.
        candidate_budgets: Exact graph token totals to compare.
        cost_table: Measured whole-step cost surface.
        prefix_mask_steps: Optional observed prefix-acceptance labels matching
            ``survival_steps``. When present, confidence determines allocation
            order while labels determine realized accepted-token yield.

    Returns:
        The selected V and the estimated goodput of every evaluated V.
    """
    values = np.asarray(survival_steps, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"survival_steps must have shape [steps, G, K], got {values.shape}")
    num_steps, num_requests, max_draft_len = values.shape
    if num_steps == 0 or num_requests == 0 or max_draft_len == 0:
        raise ValueError("survival_steps must have non-zero dimensions")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("survival_steps entries must be finite probabilities")
    labels = None
    if prefix_mask_steps is not None:
        labels = np.asarray(prefix_mask_steps, dtype=np.float64)
        if labels.shape != values.shape:
            raise ValueError(
                "prefix_mask_steps must have the same shape as survival_steps; "
                f"got {labels.shape} and {values.shape}"
            )
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("prefix_mask_steps entries must be binary")
        if np.any(labels[:, :, 1:] > labels[:, :, :-1]):
            raise ValueError("prefix_mask_steps rows must be prefix-closed")

    minimum = num_requests
    full = num_requests * (max_draft_len + 1)
    budgets = sorted({int(value) for value in candidate_budgets} | {full})
    if any(value < minimum or value > full for value in budgets):
        raise ValueError(f"candidate budgets must lie in [{minimum}, {full}]")
    if cost_table.is_flat:
        return full, {full: 0.0}

    flat_scores = values.reshape(num_steps, -1)
    order = np.argsort(-flat_scores, axis=1, kind="stable")
    if labels is None:
        ordered_yield = np.take_along_axis(flat_scores, order, axis=1)
    else:
        ordered_yield = np.take_along_axis(labels.reshape(num_steps, -1), order, axis=1)
    expected_yield = float(num_requests) + np.concatenate(
        (
            np.zeros((num_steps, 1), dtype=np.float64),
            np.cumsum(ordered_yield, axis=1),
        ),
        axis=1,
    )
    scores = {
        budget: float(expected_yield[:, budget - num_requests].mean())
        / cost_table.step_time(budget, num_requests)
        for budget in budgets
    }
    best = max(budgets, key=lambda budget: (scores[budget], budget))
    full_score = scores[full]
    if best != full and scores[best] < full_score * (1.0 + cost_table.minimum_predicted_gain):
        best = full
    return best, scores
