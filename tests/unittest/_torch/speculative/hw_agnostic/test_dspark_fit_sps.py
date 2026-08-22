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
"""Tests for the offline DSpark SPS cost fitter."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[4] / "microbenchmarks/dspark_fit_sps.py"
_SPEC = importlib.util.spec_from_file_location("dspark_fit_sps_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Cell = _MODULE.Cell
aligned_step_times = _MODULE.aligned_step_times
fit_additive_cost = _MODULE.fit_additive_cost
load_cells = _MODULE.load_cells

_OPTIMIZER_PATH = Path(__file__).parents[4] / "microbenchmarks/dspark_optimize_fixed_budget.py"
_OPTIMIZER_SPEC = importlib.util.spec_from_file_location(
    "dspark_optimize_fixed_budget_test_module", _OPTIMIZER_PATH
)
assert _OPTIMIZER_SPEC is not None and _OPTIMIZER_SPEC.loader is not None
_OPTIMIZER = importlib.util.module_from_spec(_OPTIMIZER_SPEC)
_OPTIMIZER_SPEC.loader.exec_module(_OPTIMIZER)

_STS_PATH = Path(__file__).parents[4] / "microbenchmarks/dspark_fit_sts.py"
_STS_SPEC = importlib.util.spec_from_file_location("dspark_fit_sts_test_module", _STS_PATH)
assert _STS_SPEC is not None and _STS_SPEC.loader is not None
_STS = importlib.util.module_from_spec(_STS_SPEC)
_STS_SPEC.loader.exec_module(_STS)


def _cell(batch_size: int, verify_len: int) -> Cell:
    verifier_tokens = batch_size * (verify_len + 1)
    step_time = 10.0 + 0.5 * math.log2(batch_size) + 0.01 * verifier_tokens
    return Cell(
        batch_size=batch_size,
        verify_len=verify_len,
        verifier_tokens=verifier_tokens,
        step_time_ms=step_time,
        num_samples=40,
        p10_ms=step_time,
        p90_ms=step_time,
    )


def test_fit_additive_cost_recovers_connected_surface() -> None:
    cells = [
        _cell(batch_size, verify_len) for batch_size in (1, 2, 4, 8) for verify_len in (1, 3, 5)
    ]

    result = fit_additive_cost(cells)

    assert result["fit"]["max_abs_residual_ms"] < 1e-10
    assert result["token_counts"] == sorted({cell.verifier_tokens for cell in cells})
    assert all(
        right >= left for left, right in zip(result["step_time_ms"], result["step_time_ms"][1:])
    )
    assert all(value >= 0.0 for value in result["batch_overhead_ms"])


def test_fit_additive_cost_refuses_disconnected_grid() -> None:
    cells = [_cell(1, 1), _cell(2, 1)]

    with pytest.raises(ValueError, match="disconnected"):
        fit_additive_cost(cells)


def test_aligned_step_times_filters_non_decode_shapes() -> None:
    rows = [
        {
            "iter": 1,
            "hostStepTimeMS": 3.0,
            "inflightBatchingStats": {
                "numContextRequests": 0,
                "numGenRequests": 4,
            },
        },
        {
            "iter": 2,
            "hostStepTimeMS": 4.0,
            "inflightBatchingStats": {
                "numContextRequests": 1,
                "numGenRequests": 4,
            },
        },
        {
            "iter": 3,
            "hostStepTimeMS": 5.0,
            "inflightBatchingStats": {
                "numContextRequests": 0,
                "numGenRequests": 2,
            },
        },
        {
            "iter": 4,
            "hostStepTimeMS": 6.0,
            "inflightBatchingStats": {
                "numContextRequests": 0,
                "numGenRequests": 4,
            },
        },
    ]

    assert aligned_step_times(
        rows,
        batch_size=4,
        timing_key="hostStepTimeMS",
    ) == [(1, 3.0), (4, 6.0)]


def test_load_cells_combines_repeated_state_globs(tmp_path: Path) -> None:
    for name, budget, step_time in (("l1", 2, 3.0), ("l3", 4, 4.0)):
        state = tmp_path / name
        results = state / "results"
        results.mkdir(parents=True)
        (state / "starting.json").write_text(
            json.dumps(
                {
                    "model": "/models/dspark",
                    "schedule": {"1": budget},
                }
            ),
            encoding="utf-8",
        )
        (results / "0001.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "command": {"prompts": ["prompt"]},
                    "iteration_stats": [
                        {
                            "iter": 1,
                            "hostStepTimeMS": step_time,
                            "inflightBatchingStats": {
                                "numContextRequests": 0,
                                "numGenRequests": 1,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    cells, states, model = load_cells(
        state_glob=[str(tmp_path / "l1"), str(tmp_path / "l3")],
        timing_key="hostStepTimeMS",
        warmup_results=0,
        warmup_steps=0,
        min_samples=1,
    )

    assert [(cell.verify_len, cell.step_time_ms) for cell in cells] == [(1, 3.0), (3, 4.0)]
    assert states == [str(tmp_path / "l1"), str(tmp_path / "l3")]
    assert model == "/models/dspark"


def _write_optimizer_inputs(
    tmp_path: Path,
    *,
    prefix_mask: torch.Tensor,
    token_counts: list[int],
    step_time_ms: list[float],
) -> tuple[Path, Path, Path]:
    trace = tmp_path / "trace.pt"
    torch.save(
        {
            "logits": torch.zeros_like(prefix_mask, dtype=torch.float32),
            "prefix_mask": prefix_mask,
            "graph_batch_size": 1,
        },
        trace,
    )
    table = tmp_path / "sps.json"
    table.write_text(
        json.dumps(
            {
                "token_counts": token_counts,
                "step_time_ms": step_time_ms,
                "minimum_predicted_gain": 0.01,
            }
        ),
        encoding="utf-8",
    )
    sts = tmp_path / "sts.json"
    sts.write_text(
        json.dumps({"sts_temperatures": [1.0] * prefix_mask.shape[-1]}),
        encoding="utf-8",
    )
    return trace, table, sts


def test_optimizer_emits_disabled_config_when_full_k_wins(tmp_path: Path) -> None:
    trace, table, sts = _write_optimizer_inputs(
        tmp_path,
        prefix_mask=torch.ones((2, 1, 1), dtype=torch.bool),
        token_counts=[1, 2],
        step_time_ms=[1.0, 1.0],
    )

    result = _OPTIMIZER.optimize(
        trace_glob=str(trace),
        sps_table_path=str(table),
        sts_path=str(sts),
        max_candidates=4,
    )

    assert result["confidence_mode"] == "disabled"
    assert result["confidence_verifier_token_budget_schedule"] is None
    assert result["confidence_sts_path"] is None
    assert result["optimization"]["1"]["runtime_action"] == "static_full_k_fallback"


def test_optimizer_emits_only_profitable_trimmed_budget(tmp_path: Path) -> None:
    trace, table, sts = _write_optimizer_inputs(
        tmp_path,
        prefix_mask=torch.zeros((2, 1, 2), dtype=torch.bool),
        token_counts=[2, 3],
        step_time_ms=[1.0, 10.0],
    )

    result = _OPTIMIZER.optimize(
        trace_glob=str(trace),
        sps_table_path=str(table),
        sts_path=str(sts),
        max_candidates=4,
    )

    assert result["confidence_mode"] == "fixed_budget"
    assert result["confidence_verifier_token_budget_schedule"] == {"1": 2}
    assert result["confidence_sts_path"] == str(sts)
    assert result["optimization"]["1"]["runtime_action"] == "confidence_fixed_budget"


def test_optimizer_uses_exact_cost_curve_for_each_graph_batch_size(tmp_path: Path) -> None:
    for graph_batch_size in (1, 2):
        prefix_mask = torch.zeros((2, graph_batch_size, 2), dtype=torch.bool)
        torch.save(
            {
                "logits": torch.zeros_like(prefix_mask, dtype=torch.float32),
                "prefix_mask": prefix_mask,
                "graph_batch_size": graph_batch_size,
            },
            tmp_path / f"trace-g{graph_batch_size}.pt",
        )
    table = tmp_path / "multi-g-sps.json"
    table.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "minimum_predicted_gain": 0.01,
                "cost_tables": {
                    "1": {
                        "token_counts": [2, 3],
                        "step_time_ms": [1.0, 10.0],
                    },
                    "2": {
                        "token_counts": [4, 6],
                        "step_time_ms": [2.0, 20.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    sts = tmp_path / "sts.json"
    sts.write_text(json.dumps({"sts_temperatures": [1.0, 1.0]}), encoding="utf-8")

    result = _OPTIMIZER.optimize(
        trace_glob=str(tmp_path / "trace-g*.pt"),
        sps_table_path=str(table),
        sts_path=str(sts),
        max_candidates=4,
    )

    assert result["confidence_mode"] == "fixed_budget"
    assert result["confidence_verifier_token_budget_schedule"] == {"1": 2, "2": 4}
    assert result["optimization"]["1"]["selected_budget"] == 2
    assert result["optimization"]["2"]["selected_budget"] == 4


def test_sts_fitter_sanitizes_non_finite_logits_before_bce() -> None:
    logits = torch.tensor(
        [
            [float("nan"), float("inf"), float("-inf")],
            [0.0, 0.0, 0.0],
        ]
    )
    prefix_mask = torch.zeros_like(logits, dtype=torch.bool)

    result = _STS.fit_sts_temperatures(
        logits=logits,
        prefix_mask=prefix_mask,
        grid=[1.0],
        min_samples=1,
    )

    assert result["non_finite_logits_sanitized"] == 3
    assert all(math.isfinite(value) and value > 0.0 for value in result["temperatures"])
    assert all(math.isfinite(value) for value in result["ece_after"])
