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
"""Fit a DSpark verifier-cost table from persistent-harness results.

The profiler launches keep the draft block at its deployment value and change
only the exact fixed verifier budget.  This avoids attributing a shorter draft
pass to verifier savings.  Profile a connected cross product of graph batch
size and uniform retained length, for example L={1,3,5} on a power-of-two batch
ladder, then fit

    T(G, V) = fixed_overhead + batch_overhead(G) + token_cost(V).

The output is directly consumable by :mod:`dspark_optimize_fixed_budget`.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

HOST_STEP_TIME_KEY = "hostStepTimeMS"
GPU_FORWARD_TIME_KEY = "gpuForwardTimeMS"
MIN_TOKEN_COST_MS = 1e-4


@dataclass(frozen=True)
class Cell:
    """Steady-state timing summary for one exact ``(G, V)`` cell."""

    batch_size: int
    verify_len: int
    verifier_tokens: int
    step_time_ms: float
    num_samples: int
    p10_ms: float
    p90_ms: float

    def to_json(self) -> dict[str, int | float]:
        return {
            "batch_size": self.batch_size,
            "verify_len": self.verify_len,
            "verifier_tokens": self.verifier_tokens,
            "step_time_ms": round(self.step_time_ms, 6),
            "num_samples": self.num_samples,
            "p10_ms": round(self.p10_ms, 6),
            "p90_ms": round(self.p90_ms, 6),
        }


def _batch_size(command: dict[str, object]) -> int:
    prompts = command.get("prompts")
    token_ids = command.get("prompt_token_ids")
    batch = prompts if prompts is not None else token_ids
    if not isinstance(batch, list) or not batch:
        raise ValueError("result command must contain a non-empty prompt batch")
    return len(batch)


def aligned_step_times(
    rows: Sequence[dict[str, object]],
    *,
    batch_size: int,
    timing_key: str,
) -> list[tuple[int, float]]:
    """Return aligned decode-only ``(iteration, milliseconds)`` samples."""
    by_iteration: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        iteration = row.get("iter")
        if iteration is not None:
            by_iteration[int(iteration)].append(row)

    kept: list[tuple[int, float]] = []
    for iteration, group in sorted(by_iteration.items()):
        batching = [row.get("inflightBatchingStats") or {} for row in group]
        if any(int(stats.get("numContextRequests", 0) or 0) for stats in batching):
            continue
        generation_counts = {int(stats.get("numGenRequests", 0) or 0) for stats in batching}
        if generation_counts != {batch_size}:
            continue
        timings = [row.get(timing_key) for row in group]
        timings = [float(value) for value in timings if value is not None]
        if not timings or any(not math.isfinite(value) or value <= 0.0 for value in timings):
            continue
        # Attention-DP is disabled for fixed-budget validation.  If a future
        # stats producer emits duplicate rank rows, the slowest rank is the
        # conservative whole-step cost.
        kept.append((iteration, max(timings)))
    return kept


def load_cells(
    *,
    state_glob: str | Sequence[str],
    timing_key: str,
    warmup_results: int,
    warmup_steps: int,
    min_samples: int,
) -> tuple[list[Cell], list[str], str | None]:
    """Load harness states and summarize every requested exact-V cell."""
    patterns = [state_glob] if isinstance(state_glob, str) else list(state_glob)
    states = [
        Path(path) for path in sorted({path for pattern in patterns for path in glob.glob(pattern)})
    ]
    if not states:
        raise ValueError(f"No harness states matched {patterns!r}")

    samples: dict[tuple[int, int, int], list[list[tuple[int, float]]]] = defaultdict(list)
    model: str | None = None
    for state in states:
        starting_path = state / "starting.json"
        if not starting_path.exists():
            raise ValueError(f"{state} has no starting.json")
        starting = json.loads(starting_path.read_text(encoding="utf-8"))
        current_model = str(starting.get("model"))
        if model is None:
            model = current_model
        elif current_model != model:
            raise ValueError("Refusing to combine profiler states from different models")
        schedule = {int(key): int(value) for key, value in starting["schedule"].items()}
        state_samples: dict[tuple[int, int, int], list[list[tuple[int, float]]]] = defaultdict(list)
        for result_path in sorted((state / "results").glob("*.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                raise ValueError(f"Profiler command failed: {result_path}")
            batch_size = _batch_size(result["command"])
            budget = schedule.get(batch_size)
            if budget is None or budget % batch_size:
                raise ValueError(f"{state} budget {budget} is not uniform for G={batch_size}")
            verify_len = budget // batch_size - 1
            if verify_len < 0:
                raise ValueError(f"{state} has invalid budget {budget} for G={batch_size}")
            rows = result.get("iteration_stats")
            if not isinstance(rows, list):
                raise ValueError(f"{result_path} has no iteration_stats list")
            state_samples[(batch_size, verify_len, budget)].append(
                aligned_step_times(
                    rows,
                    batch_size=batch_size,
                    timing_key=timing_key,
                )
            )

        for key, result_samples in state_samples.items():
            samples[key].extend(result_samples[warmup_results:])

    cells: list[Cell] = []
    for (batch_size, verify_len, budget), result_samples in sorted(samples.items()):
        values = [value for result in result_samples for _, value in result[warmup_steps:]]
        if len(values) < min_samples:
            raise ValueError(
                f"cell (G={batch_size}, L={verify_len}, V={budget}) has "
                f"{len(values)} steady samples, need {min_samples}"
            )
        array = np.asarray(values, dtype=np.float64)
        cells.append(
            Cell(
                batch_size=batch_size,
                verify_len=verify_len,
                verifier_tokens=budget,
                step_time_ms=float(np.median(array)),
                num_samples=int(array.size),
                p10_ms=float(np.percentile(array, 10)),
                p90_ms=float(np.percentile(array, 90)),
            )
        )
    if not cells:
        raise ValueError("No complete profiler cells remained")
    return cells, [str(path) for path in states], model


def _assert_connected(cells: Sequence[Cell]) -> None:
    """Require the bipartite ``G``/``V`` measurement graph to be connected."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    for cell in cells:
        left, right = find(("g", cell.batch_size)), find(("v", cell.verifier_tokens))
        if left != right:
            parent[left] = right
    components = {find(("g", cell.batch_size)) for cell in cells}
    if len(components) != 1:
        raise ValueError(
            "Profiler grid is disconnected; add uniform retained lengths "
            "whose V=G*(L+1) values collide across adjacent batch sizes"
        )


def _running_max(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    current = -float("inf")
    for value in values:
        current = max(current, float(value))
        result.append(current)
    return result


def _zero_intercept(xs: Sequence[int], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or np.ptp(x) == 0.0:
        return float(y.min())
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    return float(y.mean() - slope * x.mean())


def fit_additive_cost(cells: Sequence[Cell]) -> dict[str, object]:
    """Fit and conservatively monotonicize the measured additive model."""
    _assert_connected(cells)
    batch_sizes = sorted({cell.batch_size for cell in cells})
    token_counts = sorted({cell.verifier_tokens for cell in cells})
    batch_index = {value: index for index, value in enumerate(batch_sizes)}
    token_index = {value: index for index, value in enumerate(token_counts)}
    design = np.zeros((len(cells), len(batch_sizes) + len(token_counts)), dtype=np.float64)
    observed = np.empty(len(cells), dtype=np.float64)
    for row, cell in enumerate(cells):
        design[row, batch_index[cell.batch_size]] = 1.0
        design[row, len(batch_sizes) + token_index[cell.verifier_tokens]] = 1.0
        observed[row] = cell.step_time_ms
    solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ solution
    residual = observed - predicted
    intercept = {batch_size: float(solution[index]) for batch_size, index in batch_index.items()}
    theta_raw = {
        tokens: float(solution[len(batch_sizes) + index]) for tokens, index in token_index.items()
    }
    theta_values = _running_max([theta_raw[tokens] for tokens in token_counts])
    theta_was_clamped = any(
        abs(theta_raw[token] - monotone) > 1e-9
        for token, monotone in zip(token_counts, theta_values)
    )
    shift = _zero_intercept(token_counts, theta_values)
    shift = min(shift, min(theta_values) - MIN_TOKEN_COST_MS)
    theta_values = [value - shift for value in theta_values]
    intercept_values = [intercept[batch_size] + shift for batch_size in batch_sizes]

    warnings: list[str] = []
    if min(intercept_values) < 0.0:
        warnings.append("The fitted non-verifier floor was negative and was clamped to zero")
        intercept_values = [max(value, 0.0) for value in intercept_values]
    monotone_intercepts = _running_max(intercept_values)
    if any(abs(left - right) > 1e-9 for left, right in zip(monotone_intercepts, intercept_values)):
        warnings.append("Batch overhead was not monotonic and was clamped upward")
    if theta_was_clamped:
        warnings.append("Token cost was not monotonic and was clamped upward")

    fixed = min(monotone_intercepts)
    typical = float(np.median(observed))
    max_abs_residual = float(np.max(np.abs(residual)))
    return {
        "token_counts": token_counts,
        "step_time_ms": [round(value, 6) for value in theta_values],
        "fixed_overhead_ms": round(fixed, 6),
        "batch_sizes": batch_sizes,
        "batch_overhead_ms": [round(value - fixed, 6) for value in monotone_intercepts],
        "fit": {
            "max_abs_residual_ms": max_abs_residual,
            "max_rel_residual": max_abs_residual / typical if typical > 0.0 else 0.0,
            "warnings": warnings,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a decomposed DSpark SPS cost table from harness states."
    )
    parser.add_argument(
        "--state-glob",
        action="append",
        required=True,
        help="Harness-state glob. Repeat to combine disjoint profiler runs.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--timing-key",
        choices=(HOST_STEP_TIME_KEY, GPU_FORWARD_TIME_KEY),
        default=HOST_STEP_TIME_KEY,
    )
    parser.add_argument("--warmup-results", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--minimum-predicted-gain", type=float, default=0.01)
    args = parser.parse_args()
    cells, states, model = load_cells(
        state_glob=args.state_glob,
        timing_key=args.timing_key,
        warmup_results=args.warmup_results,
        warmup_steps=args.warmup_steps,
        min_samples=args.min_samples,
    )
    payload = fit_additive_cost(cells)
    payload["minimum_predicted_gain"] = args.minimum_predicted_gain
    payload["_meta"] = {
        "encoding": "decomposed",
        "lookup": "interp",
        "engine": model,
        "timing_key": args.timing_key,
        "warmup_results_per_state": args.warmup_results,
        "warmup_steps_per_result": args.warmup_steps,
        "minimum_samples_per_cell": args.min_samples,
        "states": states,
        "cells": [cell.to_json() for cell in cells],
        **payload.pop("fit"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["_meta"], indent=2))
    print(f"Wrote DSpark SPS cost table -> {output}")


if __name__ == "__main__":
    main()
