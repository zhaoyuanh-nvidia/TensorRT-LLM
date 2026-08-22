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
"""Derive a per-batch exact-V DSpark schedule from measured traces."""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tensorrt_llm._torch.speculative.dspark_planner import (
    ExactSpsCostTable,
    derive_fixed_verifier_budget_candidates,
    load_sps_cost_table,
    select_fixed_verifier_budget_from_traces,
)

TraceStep = tuple[np.ndarray, np.ndarray]


def _append_ring_steps(
    *,
    path: str,
    logits: torch.Tensor,
    prefix_mask: torch.Tensor,
    graph_batch_sizes: torch.Tensor,
    step_ids: torch.Tensor,
    grouped: dict[int, list[TraceStep]],
) -> None:
    if graph_batch_sizes.ndim != 1 or step_ids.ndim != 1:
        raise ValueError(f"{path} ring metadata graph_batch_size and step_id must be vectors")
    if logits.shape[0] != graph_batch_sizes.numel() or logits.shape[0] != step_ids.numel():
        raise ValueError(f"{path} ring metadata lengths disagree with logits")
    if prefix_mask.shape != logits.shape:
        raise ValueError(f"{path} prefix_mask shape disagrees with logits")

    keys = torch.stack((graph_batch_sizes.to(torch.int64), step_ids.to(torch.int64)), dim=1)
    for graph_batch_size, step_id in torch.unique(keys, dim=0).tolist():
        select = (keys[:, 0] == graph_batch_size) & (keys[:, 1] == step_id)
        step_logits = logits[select]
        if step_logits.shape[0] != graph_batch_size:
            # New/recycled slots and a ring wrap can leave partial events. They
            # are intentionally excluded because the fixed-V objective is
            # defined over one complete CUDA-graph batch.
            continue
        grouped[int(graph_batch_size)].append(
            (
                np.asarray(step_logits, dtype=np.float64),
                np.asarray(prefix_mask[select], dtype=np.float64),
            )
        )


def _load_temperatures(path: str | None, max_draft_len: int) -> np.ndarray:
    if path is None:
        return np.ones(max_draft_len, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("sts_temperatures", payload.get("temperatures"))
    if values is None or len(values) != max_draft_len:
        raise ValueError(f"{path} must contain {max_draft_len} STS temperatures")
    temperatures = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("STS temperatures must be positive and finite")
    return temperatures


def _load_trace_steps(data_glob: str) -> dict[int, list[TraceStep]]:
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise ValueError(f"No confidence trace shards matched {data_glob!r}")
    grouped: dict[int, list[TraceStep]] = defaultdict(list)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        logits = payload.get("logits")
        prefix_mask = payload.get("prefix_mask")
        graph_batch_size = payload.get("graph_batch_size")
        if not isinstance(logits, torch.Tensor) or logits.ndim not in (2, 3):
            raise ValueError(f"{path} must contain [G,K] or [steps,G,K] logits")
        if not isinstance(prefix_mask, torch.Tensor) or prefix_mask.shape != logits.shape:
            raise ValueError(f"{path} must contain prefix_mask matching logits")
        step_ids = payload.get("step_id")
        if (
            logits.ndim == 2
            and isinstance(graph_batch_size, torch.Tensor)
            and isinstance(step_ids, torch.Tensor)
        ):
            _append_ring_steps(
                path=path,
                logits=logits,
                prefix_mask=prefix_mask,
                graph_batch_sizes=graph_batch_size,
                step_ids=step_ids,
                grouped=grouped,
            )
            continue
        if graph_batch_size is None:
            graph_batch_size = int(logits.shape[-2])
        graph_batch_size = int(graph_batch_size)
        if logits.shape[-2] != graph_batch_size:
            raise ValueError(
                f"{path} graph_batch_size={graph_batch_size} disagrees with "
                f"logits shape {tuple(logits.shape)}"
            )
        steps = logits.unsqueeze(0) if logits.ndim == 2 else logits
        masks = prefix_mask.unsqueeze(0) if prefix_mask.ndim == 2 else prefix_mask
        grouped[graph_batch_size].extend(
            (np.asarray(step), np.asarray(mask)) for step, mask in zip(steps, masks)
        )
    if not grouped:
        raise ValueError("No complete confidence trace steps remained after loading shards")
    return grouped


def optimize(
    *,
    trace_glob: str,
    sps_table_path: str,
    sts_path: str | None,
    max_candidates: int,
) -> dict[str, object]:
    """Return a runtime config fragment and its predicted per-BS economics."""
    table, table_payload = load_sps_cost_table(sps_table_path)
    grouped = _load_trace_steps(trace_glob)
    trimmed_schedule: dict[str, int] = {}
    details: dict[str, object] = {}
    for graph_batch_size, trace_steps in sorted(grouped.items()):
        graph_cost_table = (
            table.for_graph_batch_size(graph_batch_size)
            if isinstance(table, ExactSpsCostTable)
            else table
        )
        logits = np.stack([step[0] for step in trace_steps])
        prefix_mask = np.stack([step[1] for step in trace_steps])
        max_draft_len = int(logits.shape[-1])
        temperatures = _load_temperatures(sts_path, max_draft_len)
        scaled_logits = np.nan_to_num(
            logits / temperatures,
            nan=-80.0,
            posinf=80.0,
            neginf=-80.0,
        )
        scaled_logits = np.clip(scaled_logits, -80.0, 80.0)
        conditional = 1.0 / (1.0 + np.exp(-scaled_logits))
        survival = np.cumprod(conditional, axis=2)
        candidates = derive_fixed_verifier_budget_candidates(
            cost_table=graph_cost_table,
            num_requests=graph_batch_size,
            max_draft_len=max_draft_len,
            max_candidates=max_candidates,
        )
        selected, scores = select_fixed_verifier_budget_from_traces(
            survival_steps=survival,
            candidate_budgets=candidates,
            cost_table=graph_cost_table,
            prefix_mask_steps=prefix_mask,
        )
        full = graph_batch_size * (max_draft_len + 1)
        full_score = scores.get(full, 0.0)
        predicted_gain = scores[selected] / full_score - 1.0 if full_score > 0.0 else 0.0
        if selected != full:
            trimmed_schedule[str(graph_batch_size)] = selected
        details[str(graph_batch_size)] = {
            "non_finite_logits_sanitized": int(np.count_nonzero(~np.isfinite(logits))),
            "num_trace_steps": int(logits.shape[0]),
            "candidate_budgets": list(candidates),
            "selected_budget": selected,
            "full_budget": full,
            "estimated_goodput_gain": predicted_gain,
            "estimated_goodput": {str(key): value for key, value in scores.items()},
            "objective_basis": "observed_prefix_acceptance",
            "runtime_action": (
                "confidence_fixed_budget" if selected != full else "static_full_k_fallback"
            ),
        }
    confidence_enabled = bool(trimmed_schedule)
    return {
        "confidence_mode": "fixed_budget" if confidence_enabled else "disabled",
        "confidence_verifier_token_budget_schedule": trimmed_schedule or None,
        "confidence_sts_path": sts_path if confidence_enabled else None,
        "optimization": details,
        "sps_table": sps_table_path,
        "sps_engine": (table_payload.get("_meta") or {}).get("engine"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Choose one exact verifier budget per CUDA-graph batch size from "
            "full-K confidence traces and a measured SPS cost table."
        )
    )
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--sps-table", required=True)
    parser.add_argument("--sts-path")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = optimize(
        trace_glob=args.trace_glob,
        sps_table_path=args.sps_table,
        sts_path=args.sts_path,
        max_candidates=args.max_candidates,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["optimization"], indent=2))
    print(f"Wrote DSpark fixed-budget schedule -> {output}")


if __name__ == "__main__":
    main()
