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
"""Tests for fixed- and dynamic-budget DSpark confidence planning."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_confidence import (
    apply_sts,
    build_confidence_device_layout,
    load_sts_temperatures,
    plan_dynamic_verifier_budget,
    plan_fixed_verifier_budget,
    plan_masked_fixed_verifier_draft_lens,
    plan_uniform_floor_two_tier_verifier_budget,
    resolve_uniform_floor_two_tier,
    verify_packed_greedy,
)
from tensorrt_llm._torch.speculative.dspark_planner import (
    ExactSpsCostTable,
    SpsCostTable,
    derive_fixed_verifier_budget_candidates,
    load_engine_fingerprint,
    load_sps_cost_table,
    select_fixed_verifier_budget,
    select_fixed_verifier_budget_from_traces,
    validate_sps_cost_table_payload,
)
from tensorrt_llm._torch.speculative.dspark_trace import (
    DYNAMIC_CONFIDENCE_TRACE_PATH_ENV,
    ConfidenceTraceRing,
    dynamic_confidence_trace_path_from_env,
)


@dataclass
class _ReferencePlan:
    retained_lens: list[int]
    query_lens: list[int]
    cu_query_lens: list[int]
    packed_to_dense: list[int]
    dense_to_packed: list[int]


def _reference_plan(confidence_logits: torch.Tensor, verifier_token_budget: int) -> _ReferencePlan:
    num_requests, max_draft_len = confidence_logits.shape
    prefix_scores = torch.cumprod(torch.sigmoid(confidence_logits.float()), dim=1)
    candidates = [
        (float(prefix_scores[request_id, position]), position, request_id)
        for position in range(max_draft_len)
        for request_id in range(num_requests)
    ]
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

    retained_lens = [0] * num_requests
    num_retained = verifier_token_budget - num_requests
    for _, _, request_id in candidates[:num_retained]:
        retained_lens[request_id] += 1

    query_lens = [retained_len + 1 for retained_len in retained_lens]
    cu_query_lens = [0]
    for query_len in query_lens:
        cu_query_lens.append(cu_query_lens[-1] + query_len)

    dense_width = max_draft_len + 1
    dense_capacity = num_requests * dense_width
    packed_to_dense = [
        request_id * dense_width + local_position
        for request_id, query_len in enumerate(query_lens)
        for local_position in range(query_len)
    ]
    dense_to_packed = [-1] * dense_capacity
    for packed_position, dense_position in enumerate(packed_to_dense):
        dense_to_packed[dense_position] = packed_position

    return _ReferencePlan(
        retained_lens=retained_lens,
        query_lens=query_lens,
        cu_query_lens=cu_query_lens,
        packed_to_dense=packed_to_dense,
        dense_to_packed=dense_to_packed,
    )


def _assert_matches_reference(confidence_logits: torch.Tensor, verifier_token_budget: int) -> None:
    actual = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget)
    expected = _reference_plan(confidence_logits.cpu(), verifier_token_budget)

    assert actual.retained_lens.cpu().tolist() == expected.retained_lens
    assert actual.query_lens.cpu().tolist() == expected.query_lens
    assert actual.cu_query_lens.cpu().tolist() == expected.cu_query_lens
    assert actual.packed_to_dense.cpu().tolist() == expected.packed_to_dense
    assert actual.dense_to_packed.cpu().tolist() == expected.dense_to_packed

    assert int(actual.query_lens.sum()) == verifier_token_budget
    assert actual.packed_to_dense.numel() == verifier_token_budget
    assert actual.dense_to_packed.numel() == confidence_logits.numel() + confidence_logits.shape[0]

    dense_width = confidence_logits.shape[1] + 1
    assert torch.equal(
        actual.packed_request_ids,
        torch.div(actual.packed_to_dense, dense_width, rounding_mode="floor"),
    )
    assert torch.equal(
        actual.packed_local_positions,
        torch.remainder(actual.packed_to_dense, dense_width),
    )


@pytest.mark.parametrize(
    ("logit", "expected_budget"),
    [(-20.0, 2), (20.0, 6)],
)
def test_dynamic_budget_selects_measured_goodput_tier(
    logit: float,
    expected_budget: int,
) -> None:
    logits = torch.full((2, 2), logit, dtype=torch.float32)
    budgets = torch.tensor([2, 4, 6], dtype=torch.int64)
    step_times = torch.tensor([1.0, 1.2, 1.4], dtype=torch.float32)

    plan = plan_dynamic_verifier_budget(logits, budgets, step_times)

    assert int(plan.verifier_token_budget) == expected_budget
    assert int(plan.retained_lens.sum()) == expected_budget - logits.shape[0]


def test_dynamic_budget_guard_falls_back_to_full_k() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    budgets = torch.tensor([2, 6], dtype=torch.int64)
    step_times = torch.tensor([1.0, 1.8], dtype=torch.float32)

    unguarded = plan_dynamic_verifier_budget(
        logits, budgets, step_times, minimum_predicted_gain=0.01
    )
    guarded = plan_dynamic_verifier_budget(logits, budgets, step_times, minimum_predicted_gain=0.05)

    assert int(unguarded.verifier_token_budget) == 2
    assert int(guarded.verifier_token_budget) == 6


def test_dynamic_single_tier_skips_redundant_yield_reducer() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.25]], dtype=torch.float32)
    budgets = torch.tensor([6], dtype=torch.int64)
    step_times = torch.tensor([1.4], dtype=torch.float32)

    def fail_if_called(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("a one-tier ladder cannot need cross-rank yield reduction")

    plan = plan_dynamic_verifier_budget(
        logits,
        budgets,
        step_times,
        candidate_yield_reducer=fail_if_called,
    )

    assert int(plan.verifier_token_budget) == 6
    assert int(plan.retained_lens.sum()) == 4


def test_dynamic_budget_weak_local_rank_vetoes_favorable_peer_sum() -> None:
    budgets = torch.tensor([2, 4, 6], dtype=torch.int64)
    step_times = torch.tensor([1.0, 1.1, 1.2], dtype=torch.float32)
    logits = torch.full((2, 2), 20.0, dtype=torch.float32)
    observed: list[torch.Tensor] = []

    def favorable_peer_sum(local: torch.Tensor) -> torch.Tensor:
        observed.append(local.clone())
        return local + torch.tensor([1000.0, 1000.0, 0.0])

    plan = plan_dynamic_verifier_budget(
        logits,
        budgets,
        step_times,
        candidate_yield_reducer=favorable_peer_sum,
    )

    assert torch.isneginf(observed[0][0])
    assert torch.isneginf(observed[0][1])
    assert torch.isfinite(observed[0][2])
    assert int(plan.verifier_token_budget) == 6
    assert plan.retained_lens.tolist() == [2, 2]


def test_dynamic_budget_all_local_clear_keeps_compact_candidate() -> None:
    observed: list[torch.Tensor] = []

    def all_rank_sum(local: torch.Tensor) -> torch.Tensor:
        observed.append(local.clone())
        return local * 2.0

    plan = plan_dynamic_verifier_budget(
        torch.full((2, 2), -100.0),
        torch.tensor([2, 6]),
        torch.tensor([1.0, 2.0]),
        candidate_yield_reducer=all_rank_sum,
    )

    assert torch.isfinite(observed[0]).all()
    assert int(plan.verifier_token_budget) == 2
    assert plan.retained_lens.tolist() == [0, 0]


def test_dynamic_budget_local_veto_accepts_exact_minimum_gain_threshold() -> None:
    observed: list[torch.Tensor] = []

    def capture(local: torch.Tensor) -> torch.Tensor:
        observed.append(local.clone())
        return local

    plan = plan_dynamic_verifier_budget(
        torch.full((2, 2), -100.0),
        torch.tensor([2, 6]),
        torch.tensor([1.0, 1.03]),
        minimum_predicted_gain=0.03,
        candidate_yield_reducer=capture,
    )

    assert torch.isfinite(observed[0][0])
    assert int(plan.verifier_token_budget) == 2


def test_dynamic_budget_local_veto_never_invalidates_full_candidate() -> None:
    observed: list[torch.Tensor] = []

    def capture(local: torch.Tensor) -> torch.Tensor:
        observed.append(local.clone())
        return local

    plan = plan_dynamic_verifier_budget(
        torch.full((2, 2), 20.0),
        torch.tensor([2, 4, 6]),
        torch.tensor([1.0, 1.1, 1.2]),
        candidate_yield_reducer=capture,
    )

    assert torch.isfinite(observed[0][-1])
    assert int(plan.verifier_token_budget) == 6
    assert plan.retained_lens.tolist() == [2, 2]


def test_dynamic_budget_reducer_none_keeps_existing_local_selection() -> None:
    plan = plan_dynamic_verifier_budget(
        torch.full((2, 2), -100.0),
        torch.tensor([2, 6]),
        torch.tensor([1.0, 2.0]),
        candidate_yield_reducer=None,
    )

    assert int(plan.verifier_token_budget) == 2
    assert plan.retained_lens.tolist() == [0, 0]


def test_dynamic_budget_reducer_must_preserve_candidate_shape() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    budgets = torch.tensor([2, 4, 6], dtype=torch.int64)
    step_times = torch.tensor([1.0, 1.2, 1.4], dtype=torch.float32)

    with pytest.raises(ValueError, match="preserve the candidate shape"):
        plan_dynamic_verifier_budget(
            logits,
            budgets,
            step_times,
            candidate_yield_reducer=lambda values: values[:2],
        )


def test_dynamic_budget_mask_excludes_padding_rows_from_allocation() -> None:
    logits = torch.full((4, 2), 20.0, dtype=torch.float32)
    budgets = torch.tensor([4, 6, 12], dtype=torch.int64)
    step_times = torch.tensor([1.5, 1.2, 10.0], dtype=torch.float32)
    real_request_mask = torch.tensor([True, True, False, False])

    plan = plan_dynamic_verifier_budget(
        logits,
        budgets,
        step_times,
        real_request_mask=real_request_mask,
    )

    assert int(plan.verifier_token_budget) == 6
    assert plan.retained_lens.tolist() == [1, 1, 0, 0]


def test_dynamic_budget_mask_invalidates_tier_before_yield_reduction() -> None:
    logits = torch.full((4, 2), 20.0, dtype=torch.float32)
    budgets = torch.tensor([8, 12], dtype=torch.int64)
    step_times = torch.tensor([0.1, 100.0], dtype=torch.float32)
    real_request_mask = torch.tensor([True, False, False, False])
    observed: list[torch.Tensor] = []

    def capture(values: torch.Tensor) -> torch.Tensor:
        observed.append(values.clone())
        return values

    plan = plan_dynamic_verifier_budget(
        logits,
        budgets,
        step_times,
        candidate_yield_reducer=capture,
        real_request_mask=real_request_mask,
    )

    assert torch.isneginf(observed[0][0])
    assert torch.isfinite(observed[0][1])
    assert int(plan.verifier_token_budget) == 12
    assert plan.retained_lens.tolist() == [2, 0, 0, 0]


def test_dynamic_budget_peer_ineligibility_forces_shared_full_k() -> None:
    logits = torch.full((4, 2), 20.0, dtype=torch.float32)
    budgets = torch.tensor([8, 12], dtype=torch.int64)
    step_times = torch.tensor([1.0, 100.0], dtype=torch.float32)

    def peer_reducer(values: torch.Tensor) -> torch.Tensor:
        reduced = values.clone()
        reduced[0] = float("-inf")
        return reduced

    plan = plan_dynamic_verifier_budget(
        logits,
        budgets,
        step_times,
        candidate_yield_reducer=peer_reducer,
        real_request_mask=torch.ones(4, dtype=torch.bool),
    )

    assert int(plan.verifier_token_budget) == 12
    assert plan.retained_lens.tolist() == [2, 2, 2, 2]


def _g128_floor_logits(flat: bool = False, device: str = "cpu") -> torch.Tensor:
    logits = torch.full((128, 5), 20.0, dtype=torch.float32, device=device)
    if not flat:
        logits[96:, :] = -20.0
    return logits


@pytest.mark.parametrize(
    ("step_times", "expected_budget", "expected_lens"),
    [
        ([1.0, 100.0, 100.0], 512, [3] * 128),
        ([100.0, 1.0, 100.0], 704, [5] * 96 + [3] * 32),
        ([100.0, 100.0, 1.0], 768, [5] * 128),
    ],
)
def test_dynamic_common_floor_g128_tiers(
    step_times: list[float], expected_budget: int, expected_lens: list[int]
) -> None:
    plan = plan_dynamic_verifier_budget(
        _g128_floor_logits(),
        torch.tensor([512, 704, 768], dtype=torch.int64),
        torch.tensor(step_times, dtype=torch.float32),
        minimum_predicted_gain=0.0,
    )

    assert int(plan.verifier_token_budget) == expected_budget
    assert plan.retained_lens.tolist() == expected_lens


def test_dynamic_common_floor_flat_v704_uses_stable_position_major_tie_order() -> None:
    plan = plan_dynamic_verifier_budget(
        _g128_floor_logits(flat=True),
        torch.tensor([512, 704, 768], dtype=torch.int64),
        torch.tensor([100.0, 1.0, 100.0], dtype=torch.float32),
        minimum_predicted_gain=0.0,
        temperatures=torch.full((5,), 1.0e30, dtype=torch.float32),
    )

    assert int(plan.verifier_token_budget) == 704
    assert plan.retained_lens.tolist() == [5] * 64 + [4] * 64


def test_dynamic_common_floor_padding_and_adp_ineligibility_preserve_full_fallback() -> None:
    observed: list[torch.Tensor] = []

    def peer_reducer(values: torch.Tensor) -> torch.Tensor:
        observed.append(values.clone())
        reduced = values.clone()
        reduced[1] = float("-inf")
        return reduced

    real_request_mask = torch.zeros(128, dtype=torch.bool)
    real_request_mask[:112] = True
    plan = plan_dynamic_verifier_budget(
        _g128_floor_logits(),
        torch.tensor([512, 704, 768], dtype=torch.int64),
        torch.tensor([100.0, 1.0, 100.0], dtype=torch.float32),
        minimum_predicted_gain=0.0,
        candidate_yield_reducer=peer_reducer,
        real_request_mask=real_request_mask,
    )

    assert torch.isneginf(observed[0][1])
    assert int(plan.verifier_token_budget) == 768
    assert plan.retained_lens[:112].tolist() == [5] * 112
    assert plan.retained_lens[112:].tolist() == [0] * 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dynamic_common_floor_cuda_graph_v512_v704_v512_replay() -> None:
    logits = _g128_floor_logits(device="cuda")
    budgets = torch.tensor([512, 704, 768], dtype=torch.int64, device="cuda")
    step_times = torch.tensor([1.0, 100.0, 100.0], dtype=torch.float32, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            plan_dynamic_verifier_budget(
                logits, budgets, step_times, minimum_predicted_gain=0.0
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan = plan_dynamic_verifier_budget(
            logits, budgets, step_times, minimum_predicted_gain=0.0
        )

    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 512
    assert plan.retained_lens.tolist() == [3] * 128

    step_times.copy_(torch.tensor([100.0, 1.0, 100.0], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 704
    assert plan.retained_lens.tolist() == [5] * 96 + [3] * 32

    step_times.copy_(torch.tensor([1.0, 100.0, 100.0], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 512
    assert plan.retained_lens.tolist() == [3] * 128


def test_resolve_uniform_floor_two_tier_is_fail_closed() -> None:
    assert resolve_uniform_floor_two_tier(128, 5, [512, 768]) == 3
    assert resolve_uniform_floor_two_tier(128, 5, [512, 704, 768]) is None
    assert resolve_uniform_floor_two_tier(128, 5, [513, 768]) is None
    assert resolve_uniform_floor_two_tier(128, 5, [512, 767]) is None
    assert resolve_uniform_floor_two_tier(128, 5, [768, 768]) is None


def test_uniform_floor_two_tier_fast_path_avoids_global_ranking(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "argsort",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("argsort called")),
    )
    plan = plan_uniform_floor_two_tier_verifier_budget(
        torch.full((4, 2), -20.0),
        torch.tensor([8, 12]),
        torch.tensor([1.0, 2.0]),
        uniform_compact_floor=1,
    )
    assert int(plan.verifier_token_budget) == 8
    assert plan.retained_lens.tolist() == [1, 1, 1, 1]


def test_uniform_floor_local_min_progress_vetoes_favorable_rank_sum() -> None:
    logits = torch.full((4, 2), -20.0)
    logits[0] = 20.0
    budgets = torch.tensor([8, 12])
    times = torch.tensor([1.0, 1.5])

    no_progress = plan_uniform_floor_two_tier_verifier_budget(
        logits,
        budgets,
        times,
        uniform_compact_floor=1,
        minimum_predicted_gain=0.03,
        candidate_yield_reducer=lambda values: values,
    )
    protected_laggard = plan_uniform_floor_two_tier_verifier_budget(
        logits,
        budgets,
        times,
        uniform_compact_floor=1,
        minimum_predicted_gain=0.03,
        candidate_yield_reducer=lambda values: values,
        request_progress=torch.tensor([0, 4, 4, 4]),
    )

    assert int(no_progress.verifier_token_budget) == 8
    assert int(protected_laggard.verifier_token_budget) == 12
    assert protected_laggard.retained_lens.tolist() == [2, 2, 2, 2]


def test_uniform_floor_local_min_all_clear_keeps_compact() -> None:
    plan = plan_uniform_floor_two_tier_verifier_budget(
        torch.full((4, 2), -20.0),
        torch.tensor([8, 12]),
        torch.tensor([1.0, 1.5]),
        uniform_compact_floor=1,
        minimum_predicted_gain=0.03,
        candidate_yield_reducer=lambda values: values * 2.0,
        request_progress=torch.tensor([0, 4, 4, 4]),
    )
    assert int(plan.verifier_token_budget) == 8
    assert plan.retained_lens.tolist() == [1, 1, 1, 1]


def test_uniform_floor_padding_falls_back_to_full() -> None:
    plan = plan_uniform_floor_two_tier_verifier_budget(
        torch.full((4, 2), -20.0),
        torch.tensor([8, 12]),
        torch.tensor([1.0, 2.0]),
        uniform_compact_floor=1,
        real_request_mask=torch.tensor([True, True, True, False]),
        request_progress=torch.tensor([0, 1, 2, 0]),
    )
    assert int(plan.verifier_token_budget) == 12
    assert plan.retained_lens.tolist() == [2, 2, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uniform_floor_cuda_graph_weak_strong_weak_progress_replay() -> None:
    logits = torch.full((4, 2), -20.0, device="cuda")
    logits[0] = 20.0
    budgets = torch.tensor([8, 12], device="cuda")
    times = torch.tensor([1.0, 1.5], device="cuda")
    progress = torch.tensor([0, 4, 4, 4], device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            plan_uniform_floor_two_tier_verifier_budget(
                logits,
                budgets,
                times,
                1,
                minimum_predicted_gain=0.03,
                request_progress=progress,
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan = plan_uniform_floor_two_tier_verifier_budget(
            logits,
            budgets,
            times,
            1,
            minimum_predicted_gain=0.03,
            request_progress=progress,
        )
    lens_ptr = plan.retained_lens.data_ptr()
    budget_ptr = plan.verifier_token_budget.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 12
    assert plan.retained_lens.tolist() == [2, 2, 2, 2]

    logits[0].fill_(-20.0)
    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 8
    assert plan.retained_lens.tolist() == [1, 1, 1, 1]
    assert plan.retained_lens.data_ptr() == lens_ptr
    assert plan.verifier_token_budget.data_ptr() == budget_ptr

    logits[0].fill_(20.0)
    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 12
    assert plan.retained_lens.data_ptr() == lens_ptr
    assert plan.verifier_token_budget.data_ptr() == budget_ptr


@pytest.mark.parametrize(
    "real_request_mask",
    [
        torch.ones(4),
        torch.ones(4, dtype=torch.bool).reshape(2, 2),
        torch.ones(3, dtype=torch.bool),
    ],
)
def test_dynamic_budget_real_request_mask_must_match_requests(
    real_request_mask: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="real_request_mask"):
        plan_dynamic_verifier_budget(
            torch.zeros((4, 2)),
            torch.tensor([8, 12]),
            torch.tensor([1.0, 2.0]),
            real_request_mask=real_request_mask,
        )


@pytest.mark.parametrize("verifier_token_budget", [4, 5, 11, 24])
def test_plan_matches_cpu_reference_and_exact_budget(
    verifier_token_budget: int,
) -> None:
    confidence_logits = torch.tensor(
        [
            [2.0, 0.5, -1.0, 1.5, -0.5],
            [0.8, 1.2, 0.1, -0.7, 0.3],
            [-0.4, 2.1, 0.9, -1.2, 1.1],
            [1.3, -0.2, 1.7, 0.4, -2.0],
        ]
    )
    _assert_matches_reference(confidence_logits, verifier_token_budget)

    plan = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget)
    for retained_len in plan.retained_lens.tolist():
        assert 0 <= retained_len <= confidence_logits.shape[1]


def test_masked_fixed_lens_exclude_padding_rows_and_keep_exact_budget():
    logits = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [3.5, 2.5, 1.5, 0.5, -0.5],
            [100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ]
    )
    real_mask = torch.tensor([True, True, False, False])

    retained = plan_masked_fixed_verifier_draft_lens(
        logits,
        verifier_token_budget=7,
        real_request_mask=real_mask,
    )

    assert retained[2:].tolist() == [0, 0]
    assert int(retained.sum()) == 3
    assert int(retained.max()) <= 5


def test_masked_fixed_lens_fall_back_to_real_full_k_when_infeasible():
    logits = torch.zeros(4, 5)
    real_mask = torch.tensor([True, True, False, False])

    retained = plan_masked_fixed_verifier_draft_lens(
        logits,
        verifier_token_budget=16,
        real_request_mask=real_mask,
    )

    assert retained.tolist() == [5, 5, 0, 0]


def test_g128_v256_undercounted_mask_publishes_v0_semantics():
    logits = torch.zeros(128, 5)
    real_mask = torch.zeros(128, dtype=torch.bool)
    real_mask[:17] = True
    retained = plan_masked_fixed_verifier_draft_lens(
        logits,
        verifier_token_budget=256,
        real_request_mask=real_mask,
    )

    layout = build_confidence_device_layout(
        retained,
        real_mask,
        torch.arange(128),
        verifier_token_budget=256,
        physical_draft_len=5,
    )

    assert retained.tolist() == [5] * 17 + [0] * 111
    assert int(layout.retained_token_count) == 85
    assert int(layout.query_token_count) == 213
    assert int(layout.cu_query_token_count) == 213
    assert int(layout.real_request_count) == 17
    assert int(layout.declared_verifier_token_budget) == 256
    assert int(layout.semantic_valid) == 0
    assert int(layout.verifier_token_budget) == 0


def test_masked_fixed_lens_match_original_when_every_row_is_real():
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [1.5, 0.5, -0.5],
            [1.0, 0.0, -1.0],
        ]
    )
    budget = 8
    expected = plan_fixed_verifier_budget(logits, budget).retained_lens

    actual = plan_masked_fixed_verifier_draft_lens(
        logits,
        verifier_token_budget=budget,
        real_request_mask=torch.ones(3, dtype=torch.bool),
    )

    assert torch.equal(actual, expected)


def test_masked_fixed_k6_v816_keeps_k5_floor_and_assigns_48_extras():
    retained = plan_masked_fixed_verifier_draft_lens(
        torch.zeros(128, 6),
        verifier_token_budget=816,
        real_request_mask=torch.ones(128, dtype=torch.bool),
    )

    assert retained.tolist() == [6] * 48 + [5] * 80
    assert int(retained.sum()) == 688
    assert int(retained.min()) == 5


def test_masked_fixed_k6_confidence_ranks_only_sixth_draft():
    logits = torch.zeros(4, 6)
    logits[:, -1] = torch.tensor([-10.0, 10.0, -5.0, 5.0])

    retained = plan_masked_fixed_verifier_draft_lens(
        logits,
        verifier_token_budget=26,
        real_request_mask=torch.ones(4, dtype=torch.bool),
    )

    assert retained.tolist() == [5, 6, 5, 6]


def test_masked_fixed_full_k_handles_empty_ranked_suffix():
    retained = plan_masked_fixed_verifier_draft_lens(
        torch.zeros(4, 6),
        verifier_token_budget=28,
        real_request_mask=torch.ones(4, dtype=torch.bool),
    )

    assert retained.tolist() == [6] * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_masked_fixed_k6_v816_cuda_graph_replay_preserves_floor_and_storage():
    logits = torch.zeros(128, 6, device="cuda")
    real_mask = torch.ones(128, dtype=torch.bool, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            plan_masked_fixed_verifier_draft_lens(logits, 816, real_mask)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retained = plan_masked_fixed_verifier_draft_lens(logits, 816, real_mask)
    retained_ptr = retained.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    assert retained.tolist() == [6] * 48 + [5] * 80

    logits[:, -1].copy_(torch.linspace(-10.0, 10.0, 128, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert retained.tolist() == [5] * 80 + [6] * 48
    assert int(retained.sum()) == 688
    assert int(retained.min()) == 5
    assert retained.data_ptr() == retained_ptr


def test_randomized_plans_match_cpu_reference_and_preserve_invariants() -> None:
    generator = torch.Generator().manual_seed(20260818)
    for num_requests in (1, 2, 7, 128):
        for max_draft_len in (1, 3, 5):
            logits = torch.randn(
                num_requests,
                max_draft_len,
                generator=generator,
            )
            full = num_requests * (max_draft_len + 1)
            budgets = {
                num_requests,
                min(full, num_requests + 1),
                (num_requests + full) // 2,
                full,
            }
            for budget in sorted(budgets):
                _assert_matches_reference(logits, budget)
                plan = plan_fixed_verifier_budget(logits, budget)
                assert int(plan.query_lens.sum()) == budget
                assert int(plan.retained_lens.sum()) == budget - num_requests
                assert torch.all(plan.retained_lens >= 0)
                assert torch.all(plan.retained_lens <= max_draft_len)
                assert torch.equal(
                    plan.cu_query_lens[1:] - plan.cu_query_lens[:-1],
                    plan.query_lens,
                )


def test_eight_rank_reduced_yield_selects_one_identical_tier() -> None:
    generator = torch.Generator().manual_seed(8)
    rank_logits = [torch.randn(4, 5, generator=generator) for _ in range(8)]
    budgets = torch.tensor([8, 16, 24], dtype=torch.int64)
    step_times = torch.tensor([1.0, 1.25, 1.65], dtype=torch.float32)
    rank_yields: list[torch.Tensor] = []

    def make_capture(sink: list[torch.Tensor]):
        def capture(values: torch.Tensor) -> torch.Tensor:
            sink.append(values.clone())
            return values

        return capture

    for logits in rank_logits:
        captured: list[torch.Tensor] = []
        plan_dynamic_verifier_budget(
            logits,
            budgets,
            step_times,
            candidate_yield_reducer=make_capture(captured),
        )
        rank_yields.append(captured[0])

    global_yield = torch.stack(rank_yields).sum(dim=0)
    selected = []
    for logits in rank_logits:
        plan = plan_dynamic_verifier_budget(
            logits,
            budgets,
            step_times,
            candidate_yield_reducer=lambda _local: global_yield,
        )
        selected.append(int(plan.verifier_token_budget))

    assert len(set(selected)) == 1
    expected = int(budgets[torch.argmax(global_yield / step_times)])
    assert selected == [expected] * 8


def test_equal_scores_prefer_earlier_positions_then_requests() -> None:
    num_requests, max_draft_len = 3, 3
    confidence_logits = torch.full((num_requests, max_draft_len), 1000.0)
    verifier_token_budget = num_requests + 4

    plan = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget)

    assert plan.retained_lens.tolist() == [2, 1, 1]
    _assert_matches_reference(confidence_logits, verifier_token_budget)


def test_retained_allocations_are_prefix_closed() -> None:
    confidence_logits = torch.tensor(
        [
            [8.0, -8.0, 8.0, 8.0],
            [0.5, 0.5, 0.5, 0.5],
            [-1.0, 3.0, -2.0, 4.0],
        ]
    )
    plan = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget=8)
    dense_width = confidence_logits.shape[1] + 1
    selected_drafts = plan.dense_to_packed.view(-1, dense_width)[:, 1:] >= 0

    assert not torch.any(~selected_drafts[:, :-1] & selected_drafts[:, 1:])


def test_c256_fixed_budget_has_constant_map_sizes() -> None:
    num_requests = 256
    max_draft_len = 5
    verifier_token_budget = 1152
    generator = torch.Generator().manual_seed(7)
    confidence_logits = torch.randn(num_requests, max_draft_len, generator=generator)

    plan = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget)

    assert plan.query_lens.sum().item() == verifier_token_budget
    assert plan.packed_to_dense.shape == (verifier_token_budget,)
    assert plan.dense_to_packed.shape == (num_requests * (max_draft_len + 1),)


@pytest.mark.parametrize("verifier_token_budget", [2, 16])
def test_budget_bounds(verifier_token_budget: int) -> None:
    confidence_logits = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="verifier_token_budget"):
        plan_fixed_verifier_budget(confidence_logits, verifier_token_budget)


def test_logits_must_be_a_non_empty_matrix() -> None:
    with pytest.raises(ValueError, match="shape"):
        plan_fixed_verifier_budget(torch.zeros(2, 3, 4), 2)
    with pytest.raises(ValueError, match="non-zero"):
        plan_fixed_verifier_budget(torch.zeros(0, 4), 0)


def test_full_static_budget_is_identity_mapping() -> None:
    num_requests, max_draft_len = 4, 5
    confidence_logits = torch.randn(num_requests, max_draft_len)
    verifier_token_budget = num_requests * (max_draft_len + 1)

    plan = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget)

    assert plan.retained_lens.tolist() == [max_draft_len] * num_requests
    assert torch.equal(plan.packed_to_dense, torch.arange(verifier_token_budget))
    assert torch.equal(plan.dense_to_packed, torch.arange(verifier_token_budget))


def test_verify_packed_greedy_handles_variable_prefixes() -> None:
    # d=[2, 0, 3]. Each target segment contains d verifier predictions and
    # then the golden token.
    draft_lens = torch.tensor([2, 0, 3], dtype=torch.int32)
    packed_drafts = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    target_tokens = torch.tensor([10, 99, 12, 7, 20, 21, 98, 23], dtype=torch.int32)

    accepted, accepted_lens = verify_packed_greedy(
        target_tokens, packed_drafts, draft_lens, max_draft_len=3
    )

    assert accepted_lens.tolist() == [2, 1, 3]
    assert accepted.tolist() == [
        [10, 99, 12, 0],
        [7, 0, 0, 0],
        [20, 21, 98, 23],
    ]


def test_verify_packed_greedy_supports_anchor_only_budget() -> None:
    draft_lens = torch.zeros(3, dtype=torch.int32)
    target_tokens = torch.tensor([4, 5, 6], dtype=torch.int32)

    accepted, accepted_lens = verify_packed_greedy(
        target_tokens,
        torch.empty(0, dtype=torch.int32),
        draft_lens,
        max_draft_len=5,
    )

    assert accepted_lens.tolist() == [1, 1, 1]
    assert accepted[:, 0].tolist() == [4, 5, 6]
    assert torch.count_nonzero(accepted[:, 1:]) == 0


def test_fixed_budget_plan_pack_verify_and_rewind_integration() -> None:
    confidence_logits = torch.tensor(
        [
            [8.0, 8.0, 8.0],
            [3.0, -8.0, -8.0],
            [-8.0, -8.0, -8.0],
        ]
    )
    plan = plan_fixed_verifier_budget(confidence_logits, verifier_token_budget=8)
    assert plan.retained_lens.tolist() == [3, 1, 1]

    # The physical proposal storage remains [G, K]. Only the verifier input is
    # packed to fixed V according to the retained prefixes.
    physical_drafts = torch.tensor([[10, 42, 12], [20, 91, 92], [99, 81, 82]], dtype=torch.int32)
    dense_target_tokens = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32,
    )
    packed_target_tokens = dense_target_tokens.reshape(-1).index_select(0, plan.packed_to_dense)
    packed_draft_tokens = torch.cat(
        [
            physical_drafts[request_id, :retained_len]
            for request_id, retained_len in enumerate(plan.retained_lens.tolist())
        ]
    )

    accepted, accepted_lens = verify_packed_greedy(
        packed_target_tokens,
        packed_draft_tokens,
        plan.retained_lens,
        max_draft_len=3,
    )
    accepted_drafts = accepted_lens - 1
    rewind_lens = plan.retained_lens - accepted_drafts

    assert packed_target_tokens.shape == (8,)
    assert physical_drafts.shape == (3, 3)
    assert accepted_lens.tolist() == [2, 2, 1]
    assert rewind_lens.tolist() == [2, 0, 1]
    assert accepted.tolist() == [
        [10, 11, 12, 13],
        [20, 21, 0, 0],
        [30, 31, 0, 0],
    ]


def test_load_sts_accepts_both_runtime_key_spellings(tmp_path) -> None:
    trtllm_path = tmp_path / "trtllm.json"
    sglang_path = tmp_path / "sglang.json"
    trtllm_path.write_text(json.dumps({"sts_temperatures": [0.5, 1.0, 2.0]}))
    sglang_path.write_text(json.dumps({"temperatures": [0.5, 1.0, 2.0]}))

    expected = torch.tensor([0.5, 1.0, 2.0])
    assert torch.equal(load_sts_temperatures(trtllm_path, 3), expected)
    assert torch.equal(load_sts_temperatures(sglang_path, 3), expected)


def test_load_sts_rejects_inconsistent_artifact_metadata(tmp_path) -> None:
    wrong_k_path = tmp_path / "wrong-k.json"
    wrong_k_path.write_text(json.dumps({"max_draft_len": 4, "sts_temperatures": [0.5, 1.0, 2.0]}))
    inconsistent_keys_path = tmp_path / "inconsistent-keys.json"
    inconsistent_keys_path.write_text(
        json.dumps(
            {
                "max_draft_len": 3,
                "sts_temperatures": [0.5, 1.0, 2.0],
                "temperatures": [0.5, 1.0, 3.0],
            }
        )
    )

    with pytest.raises(ValueError, match="artifact max_draft_len"):
        load_sts_temperatures(wrong_k_path, 3)
    with pytest.raises(ValueError, match="inconsistent"):
        load_sts_temperatures(inconsistent_keys_path, 3)


def _sps_payload() -> dict[str, object]:
    return {
        "token_counts": [8, 16, 24],
        "step_time_ms": [1.0, 1.2, 1.5],
        "engine_fingerprint": {
            "gpu": "B300",
            "gpu_count": 8,
            "global_graph_batch_size": 32,
            "max_draft_len": 5,
            "rank_local_graph_batch_size": 4,
            "runtime_snapshot": "runtime-v1",
            "source_head": "abc123",
            "topology": "TP8_EP8_attention_DP8",
        },
        "measurements": [
            {"rank_local_verifier_budget": 8},
            {"rank_local_verifier_budget": 16},
            {"rank_local_verifier_budget": 24},
        ],
    }


def _multi_g_sps_payload() -> dict[str, object]:
    fingerprint = {
        "gpu": "B300",
        "gpu_count": 8,
        "gpu_snapshot_sha256": "c" * 64,
        "global_graph_batch_sizes": [32, 64],
        "max_draft_len": 5,
        "rank_local_graph_batch_sizes": [4, 8],
        "runtime_snapshot": "runtime-v2",
        "source_head": "abc123",
        "source_diff_sha256": "a" * 64,
        "topology": "TP8_EP8_attention_DP8",
    }
    payload = {
        "schema_version": 2,
        "minimum_predicted_gain": 0.02,
        "cost_tables": {
            "4": {
                "token_counts": [8, 16, 24],
                "step_time_ms": [1.0, 1.2, 1.5],
            },
            "8": {
                "token_counts": [16, 32, 48],
                "step_time_ms": [1.5, 1.9, 2.4],
            },
        },
        "engine_fingerprint": fingerprint,
        "measurements": [
            {
                "rank_local_graph_batch_size": graph_batch_size,
                "rank_local_verifier_budget": verifier_token_budget,
                "step_time_ms": step_time,
                "source_result_sha256": "b" * 64,
            }
            for graph_batch_size, verifier_token_budgets, step_times in (
                (4, (8, 16, 24), (1.0, 1.2, 1.5)),
                (8, (16, 32, 48), (1.5, 1.9, 2.4)),
            )
            for verifier_token_budget, step_time in zip(verifier_token_budgets, step_times)
        ],
    }
    payload["engine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def test_sps_artifact_validation_accepts_exact_measured_tiers() -> None:
    fingerprint = validate_sps_cost_table_payload(
        _sps_payload(),
        verifier_token_budget_tiers={4: [8, 16, 24]},
        max_draft_len=5,
    )

    assert fingerprint["topology"] == "TP8_EP8_attention_DP8"


def test_multi_g_sps_artifact_requires_direct_exact_cells(tmp_path) -> None:
    path = tmp_path / "multi-g-costs.json"
    path.write_text(json.dumps(_multi_g_sps_payload()))

    table, payload = load_sps_cost_table(path)
    fingerprint = validate_sps_cost_table_payload(
        payload,
        verifier_token_budget_tiers={
            4: [8, 16, 24],
            8: [16, 32, 48],
        },
        max_draft_len=5,
        active_engine_fingerprint=payload["engine_fingerprint"],
    )

    assert isinstance(table, ExactSpsCostTable)
    assert table.minimum_predicted_gain == pytest.approx(0.02)
    assert table.step_time(16, 4) == pytest.approx(1.2)
    assert table.step_time(32, 8) == pytest.approx(1.9)
    assert fingerprint["rank_local_graph_batch_sizes"] == [4, 8]
    with pytest.raises(ValueError, match="no direct measurements for G=4, V=\\[12\\]"):
        table.step_time(12, 4)
    with pytest.raises(ValueError, match="no direct measurements for G=16"):
        table.step_time(32, 16)


def test_active_engine_fingerprint_artifact_is_authenticated(tmp_path) -> None:
    payload = _multi_g_sps_payload()
    path = tmp_path / "engine-provenance.json"
    path.write_text(
        json.dumps(
            {
                "engine_fingerprint": payload["engine_fingerprint"],
                "engine_fingerprint_sha256": payload["engine_fingerprint_sha256"],
            }
        )
    )
    assert load_engine_fingerprint(path) == payload["engine_fingerprint"]
    tampered = json.loads(path.read_text())
    tampered["engine_fingerprint"]["runtime_snapshot"] = "different"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fingerprint SHA256"):
        load_engine_fingerprint(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["cost_tables"]["8"].update(
                {"token_counts": [16, 48], "step_time_ms": [1.5, 2.4]}
            ),
            "missing pairs",
        ),
        (
            lambda payload: payload["engine_fingerprint"].update(
                {"rank_local_graph_batch_sizes": [4]}
            ),
            "must match exactly",
        ),
        (
            lambda payload: payload.update({"measurements": payload["measurements"][:-1]}),
            "provenance",
        ),
    ],
)
def test_multi_g_sps_artifact_rejects_incomplete_exact_coverage(
    mutation,
    message: str,
) -> None:
    payload = _multi_g_sps_payload()
    mutation(payload)
    payload["engine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            payload["engine_fingerprint"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match=message):
        validate_sps_cost_table_payload(
            payload,
            verifier_token_budget_tiers={
                4: [8, 16, 24],
                8: [16, 32, 48],
            },
            max_draft_len=5,
            active_engine_fingerprint=payload["engine_fingerprint"],
        )


@pytest.mark.parametrize(
    "field", ["source_diff_sha256", "runtime_snapshot", "gpu", "topology"]
)
def test_multi_g_sps_artifact_rejects_active_engine_mismatch(field: str) -> None:
    payload = _multi_g_sps_payload()
    active = dict(payload["engine_fingerprint"])
    active[field] = "different"
    with pytest.raises(ValueError, match="does not match active runtime"):
        validate_sps_cost_table_payload(
            payload,
            verifier_token_budget_tiers={
                4: [8, 16, 24],
                8: [16, 32, 48],
            },
            max_draft_len=5,
            active_engine_fingerprint=active,
        )


def test_multi_g_sps_artifact_rejects_cell_provenance_mismatch() -> None:
    payload = _multi_g_sps_payload()
    payload["measurements"][0]["step_time_ms"] += 0.1
    with pytest.raises(ValueError, match="do not match measurement"):
        validate_sps_cost_table_payload(
            payload,
            verifier_token_budget_tiers={
                4: [8, 16, 24],
                8: [16, 32, 48],
            },
            max_draft_len=5,
            active_engine_fingerprint=payload["engine_fingerprint"],
        )


@pytest.mark.parametrize(
    ("tiers", "max_draft_len", "message"),
    [
        ({4: [8, 12, 24]}, 5, "unmeasured tiers"),
        ({8: [16, 48]}, 5, "rank_local_graph_batch_size"),
        ({4: [8, 16, 24]}, 4, "max_draft_len"),
        ({4: [8, 16], 8: [16, 48]}, 5, "exactly one"),
    ],
)
def test_sps_artifact_validation_rejects_runtime_mismatch(
    tiers: dict[int, list[int]], max_draft_len: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_sps_cost_table_payload(
            _sps_payload(),
            verifier_token_budget_tiers=tiers,
            max_draft_len=max_draft_len,
        )


def test_sts_changes_cross_position_ranking() -> None:
    logits = torch.tensor([[2.0, 2.0], [1.5, 1.5]])
    identity = plan_fixed_verifier_budget(logits, verifier_token_budget=4)
    calibrated = plan_fixed_verifier_budget(
        logits,
        verifier_token_budget=4,
        temperatures=torch.tensor([10.0, 0.1]),
    )

    assert torch.equal(apply_sts(logits, None), torch.sigmoid(logits.float()))
    assert identity.retained_lens.tolist() != calibrated.retained_lens.tolist()


def test_sts_sanitizes_non_finite_confidence_logits() -> None:
    logits = torch.tensor([[float("nan"), float("inf"), float("-inf")]])

    probabilities = apply_sts(logits, None)
    calibrated = apply_sts(logits, torch.ones(3))

    expected = torch.tensor([[0.0, 1.0, 0.0]])
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(calibrated, expected)


def test_cost_table_selects_gain_and_falls_back_when_gain_is_small() -> None:
    survival = np.asarray(
        [
            [0.95, 0.90, 0.85, 0.80, 0.75],
            [0.80, 0.60, 0.40, 0.20, 0.10],
        ]
    )
    decisive = SpsCostTable(
        token_counts=(2, 6, 12),
        step_time_ms=(1.0, 1.1, 3.0),
        minimum_predicted_gain=0.01,
    )
    guarded = SpsCostTable(
        token_counts=(2, 6, 12),
        step_time_ms=(1.0, 1.0, 1.01),
        minimum_predicted_gain=0.05,
    )

    assert (
        select_fixed_verifier_budget(
            survival=survival,
            candidate_budgets=(6, 12),
            cost_table=decisive,
        )
        == 6
    )
    assert (
        select_fixed_verifier_budget(
            survival=survival,
            candidate_budgets=(6, 12),
            cost_table=guarded,
        )
        == 12
    )


def test_candidate_derivation_is_bounded_and_keeps_full_budget() -> None:
    table = SpsCostTable(
        token_counts=(0, 512, 768, 1024, 1280, 1536),
        step_time_ms=(1.0, 1.0, 1.4, 1.5, 2.5, 2.7),
    )

    candidates = derive_fixed_verifier_budget_candidates(
        cost_table=table,
        num_requests=256,
        max_draft_len=5,
        max_candidates=3,
    )

    assert len(candidates) <= 3
    assert 1536 in candidates
    assert all(512 <= value <= 1536 for value in candidates)


def test_trace_optimizer_uses_one_schedule_value_per_batch_size() -> None:
    survival_steps = np.asarray(
        [
            [
                [0.95, 0.90, 0.80, 0.70, 0.60],
                [0.80, 0.60, 0.40, 0.20, 0.10],
            ],
            [
                [0.90, 0.80, 0.70, 0.60, 0.50],
                [0.85, 0.70, 0.55, 0.30, 0.15],
            ],
        ]
    )
    table = SpsCostTable(
        token_counts=(2, 6, 12),
        step_time_ms=(1.0, 1.1, 3.0),
        minimum_predicted_gain=0.01,
    )

    budget, scores = select_fixed_verifier_budget_from_traces(
        survival_steps=survival_steps,
        candidate_budgets=(6, 12),
        cost_table=table,
    )

    assert budget == 6
    assert set(scores) == {6, 12}


def test_trace_optimizer_scores_observed_acceptance_under_confidence_order() -> None:
    survival_steps = np.asarray([[[0.9, 0.8]]])
    prefix_mask_steps = np.asarray([[[0.0, 0.0]]])
    table = SpsCostTable(
        token_counts=(2, 3),
        step_time_ms=(1.0, 1.2),
        minimum_predicted_gain=0.01,
    )

    probability_budget, _ = select_fixed_verifier_budget_from_traces(
        survival_steps=survival_steps,
        candidate_budgets=(2, 3),
        cost_table=table,
    )
    observed_budget, scores = select_fixed_verifier_budget_from_traces(
        survival_steps=survival_steps,
        candidate_budgets=(2, 3),
        cost_table=table,
        prefix_mask_steps=prefix_mask_steps,
    )

    assert probability_budget == 3
    assert observed_budget == 2
    assert scores[2] == pytest.approx(1.0)
    assert scores[3] == pytest.approx(1.0 / 1.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_capture_and_replay() -> None:
    num_requests, max_draft_len = 4, 5
    verifier_token_budget = num_requests + max_draft_len
    static_logits = torch.zeros(num_requests, max_draft_len, device="cuda", dtype=torch.float32)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            plan_fixed_verifier_budget(static_logits, verifier_token_budget)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_plan = plan_fixed_verifier_budget(static_logits, verifier_token_budget)

    static_logits.zero_()
    graph.replay()
    torch.cuda.synchronize()
    first_retained_lens = captured_plan.retained_lens.clone()

    second_logits = torch.full_like(static_logits, -20.0)
    second_logits[-1].fill_(20.0)
    static_logits.copy_(second_logits)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_plan(second_logits.cpu(), verifier_token_budget)
    assert captured_plan.retained_lens.cpu().tolist() == expected.retained_lens
    assert captured_plan.packed_to_dense.cpu().tolist() == expected.packed_to_dense
    assert not torch.equal(first_retained_lens, captured_plan.retained_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("num_requests", "verifier_token_budget"),
    [
        (16, 32),
        (16, 64),
        (16, 96),
        (32, 64),
        (32, 128),
        (32, 192),
        (64, 128),
        (64, 256),
        (64, 384),
        (128, 256),
        (128, 512),
        (128, 768),
    ],
)
def test_smaller_g_profile_cells_capture_exact_compact_v(
    num_requests: int, verifier_token_budget: int
) -> None:
    """Every staged T(G,V) cell can be planned inside a CUDA graph."""
    logits = torch.zeros((num_requests, 5), device="cuda", dtype=torch.float32)
    for _ in range(2):
        plan_fixed_verifier_budget(logits, verifier_token_budget)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan = plan_fixed_verifier_budget(logits, verifier_token_budget)
    graph.replay()
    torch.cuda.synchronize()
    assert tuple(plan.query_lens.shape) == (num_requests,)
    assert plan.cu_query_lens.numel() == num_requests + 1
    assert int(plan.query_lens.sum()) == verifier_token_budget
    assert int(plan.cu_query_lens[-1]) == verifier_token_budget
    assert int(plan.retained_lens.sum()) == verifier_token_budget - num_requests
    assert plan.packed_to_dense.numel() == verifier_token_budget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dynamic_budget_changes_tier_during_cuda_graph_replay() -> None:
    static_logits = torch.full((4, 5), -20.0, device="cuda")
    budgets = torch.tensor([8, 16, 24], dtype=torch.int64, device="cuda")
    step_times = torch.tensor([1.0, 1.2, 1.4], dtype=torch.float32, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            plan_dynamic_verifier_budget(static_logits, budgets, step_times)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_plan = plan_dynamic_verifier_budget(static_logits, budgets, step_times)

    graph.replay()
    torch.cuda.synchronize()
    assert int(captured_plan.verifier_token_budget) == 8
    assert int(captured_plan.retained_lens.sum()) == 4

    static_logits.fill_(20.0)
    graph.replay()
    torch.cuda.synchronize()
    assert int(captured_plan.verifier_token_budget) == 24
    assert int(captured_plan.retained_lens.sum()) == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dynamic_budget_local_veto_cuda_graph_weak_strong_weak_replay() -> None:
    logits = torch.full((2, 2), 20.0, dtype=torch.float32, device="cuda")
    budgets = torch.tensor([2, 6], dtype=torch.int64, device="cuda")
    step_times = torch.tensor([1.0, 1.5], dtype=torch.float32, device="cuda")

    def identity_sum(local: torch.Tensor) -> torch.Tensor:
        return local

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            plan_dynamic_verifier_budget(
                logits,
                budgets,
                step_times,
                candidate_yield_reducer=identity_sum,
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan = plan_dynamic_verifier_budget(
            logits,
            budgets,
            step_times,
            candidate_yield_reducer=identity_sum,
        )
    retained_ptr = plan.retained_lens.data_ptr()
    budget_ptr = plan.verifier_token_budget.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 6
    assert plan.retained_lens.tolist() == [2, 2]

    logits.fill_(-100.0)
    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 2
    assert plan.retained_lens.tolist() == [0, 0]

    logits.fill_(20.0)
    graph.replay()
    torch.cuda.synchronize()
    assert int(plan.verifier_token_budget) == 6
    assert plan.retained_lens.tolist() == [2, 2]
    assert plan.retained_lens.data_ptr() == retained_ptr
    assert plan.verifier_token_budget.data_ptr() == budget_ptr


def test_dynamic_confidence_trace_path_is_default_off(monkeypatch) -> None:
    monkeypatch.delenv(DYNAMIC_CONFIDENCE_TRACE_PATH_ENV, raising=False)
    assert dynamic_confidence_trace_path_from_env() is None

    monkeypatch.setenv(DYNAMIC_CONFIDENCE_TRACE_PATH_ENV, "  /tmp/dspark-dynamic  ")
    assert dynamic_confidence_trace_path_from_env() == "/tmp/dspark-dynamic"


def test_dynamic_confidence_trace_ring_records_policy_identity_and_progress(
    tmp_path,
) -> None:
    ring = ConfidenceTraceRing(
        path_stem=str(tmp_path / "dynamic-trace"),
        rank=3,
        num_slots=3,
        max_draft_len=3,
        scratch_slot=2,
        capacity=12,
        dynamic_diagnostic=True,
        device="cpu",
    )
    ring.bind_slot_identity(0, 101)
    ring.bind_slot_identity(1, 202)
    ring.bind_slot_identity(2, -1)
    ring.invalidate_slot(0)
    ring.bind_slot_identity(0, 101)
    slots = torch.tensor([0, 1, 2], dtype=torch.long)
    real_mask = torch.tensor([True, True, False])
    first_logits = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]
    )
    ring.record_and_update(
        slots=slots,
        confidence_logits=first_logits,
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
    )
    ring.publish_dynamic_plan(
        slots=slots,
        real_request_mask=real_mask,
        generation_progress=torch.tensor([4, 9, -1], dtype=torch.int64),
        selected_budget=torch.tensor(6, dtype=torch.int32),
        compact_budget=torch.tensor(6, dtype=torch.int32),
        full_budget=torch.tensor(12, dtype=torch.int32),
        compact_goodput=torch.tensor([1.25, 0.75, 0.0]),
        full_goodput=torch.tensor([1.0, 1.0, 0.0]),
    )

    ring.record_and_update(
        slots=slots,
        confidence_logits=torch.zeros_like(first_logits),
        num_accepted_tokens=torch.tensor([3, 2, 1], dtype=torch.int32),
    )
    ring.flush()

    payload = torch.load(ring.path, map_location="cpu", weights_only=True)
    assert payload["diagnostic_schema"] == "dspark_dynamic_policy_trace_v1"
    assert payload["rank"] == 3
    assert payload["rows_written"] == 3
    assert payload["slot_id"].tolist() == [0, 1, 2]
    assert payload["request_id"].tolist() == [101, 202, -1]
    assert payload["generation_progress"].tolist() == [4, 9, -1]
    assert payload["real_mask"].tolist() == [True, True, False]
    assert payload["accepted_token_count"].tolist() == [3, 2, 0]
    assert payload["accepted_draft_count"].tolist() == [2, 1, 0]
    assert payload["selected_budget"].tolist() == [6, 6, 6]
    assert payload["compact_budget"].tolist() == [6, 6, 6]
    assert payload["full_budget"].tolist() == [12, 12, 12]
    torch.testing.assert_close(
        payload["predicted_compact_minus_full_margin"],
        torch.tensor([0.25, -0.25, 0.0]),
    )
    assert payload["counterfactual_full_k_acceptance_available"] is False
    assert payload["counterfactual_full_k_acceptance"] is None
    assert "unavailable" in payload["acceptance_semantics"]


def test_dynamic_confidence_trace_capture_methods_have_no_host_reads() -> None:
    import inspect

    for method in (
        ConfidenceTraceRing.record_and_update,
        ConfidenceTraceRing.publish_dynamic_plan,
    ):
        source = inspect.getsource(method)
        assert ".item(" not in source
        assert ".cpu(" not in source
        assert "synchronize" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_confidence_trace_ring_captures_dynamic_replay_rows(tmp_path) -> None:
    max_draft_len = 3
    slots = torch.tensor([0, 1], dtype=torch.long, device="cuda")

    # Prime allocator/library paths outside the graph under test.
    warmup = ConfidenceTraceRing(
        path_stem=str(tmp_path / "warmup"),
        rank=0,
        num_slots=3,
        max_draft_len=max_draft_len,
        scratch_slot=2,
        capacity=16,
    )
    for _ in range(3):
        warmup.record_and_update(
            slots=slots,
            confidence_logits=torch.zeros((2, max_draft_len), device="cuda"),
            num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
        )
    warmup._flushed = True

    ring = ConfidenceTraceRing(
        path_stem=str(tmp_path / "trace"),
        rank=0,
        num_slots=3,
        max_draft_len=max_draft_len,
        scratch_slot=2,
        capacity=16,
    )
    logits_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
    ring.record_and_update(
        slots=slots,
        confidence_logits=logits_a,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
    )

    static_logits = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device="cuda")
    static_accepted = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ring.record_and_update(
            slots=slots,
            confidence_logits=static_logits,
            num_accepted_tokens=static_accepted,
        )

    graph.replay()
    torch.cuda.synchronize()

    static_logits.add_(10.0)
    static_accepted.copy_(torch.tensor([1, 3], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    ring.flush()

    payload = torch.load(ring.path, map_location="cpu", weights_only=True)
    assert payload["pairing"] == "draft_seq_ring"
    torch.testing.assert_close(
        payload["logits"], torch.cat((logits_a.cpu(), (static_logits - 10.0).cpu()))
    )
    torch.testing.assert_close(
        payload["prefix_mask"],
        torch.tensor(
            [[1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]],
            dtype=torch.float32,
        ),
    )
    assert payload["graph_batch_size"].tolist() == [2, 2, 2, 2]
    assert payload["step_id"].tolist() == [0, 0, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_confidence_trace_ring_reset_discards_capture_rows(tmp_path) -> None:
    max_draft_len = 3
    slots = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    ring = ConfidenceTraceRing(
        path_stem=str(tmp_path / "trace-reset"),
        rank=0,
        num_slots=3,
        max_draft_len=max_draft_len,
        scratch_slot=2,
        capacity=16,
    )
    static_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
    static_accepted = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    for _ in range(3):
        ring.record_and_update(
            slots=slots,
            confidence_logits=static_logits,
            num_accepted_tokens=static_accepted,
        )
    ring.reset()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ring.record_and_update(
            slots=slots,
            confidence_logits=static_logits,
            num_accepted_tokens=static_accepted,
        )

    ring.reset()
    graph.replay()
    static_accepted.copy_(torch.tensor([4, 2], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    ring.flush()

    payload = torch.load(ring.path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(payload["logits"], static_logits.cpu())
    torch.testing.assert_close(
        payload["prefix_mask"],
        torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32),
    )
    assert payload["graph_batch_size"].tolist() == [2, 2]
    assert payload["step_id"].tolist() == [0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_confidence_trace_ring_resets_inference_tensors(tmp_path) -> None:
    with torch.inference_mode():
        ring = ConfidenceTraceRing(
            path_stem=str(tmp_path / "inference-trace-reset"),
            rank=0,
            num_slots=3,
            max_draft_len=3,
            scratch_slot=2,
            capacity=16,
        )

    ring.reset()
    ring._flushed = True
    assert int(ring._counter) == 0
    assert int(ring._event_counter) == 0
    assert not bool(ring._slot_valid.any())
