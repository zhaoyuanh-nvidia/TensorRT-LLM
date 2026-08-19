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

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_confidence import (
    apply_sts,
    load_sts_temperatures,
    plan_dynamic_verifier_budget,
    plan_fixed_verifier_budget,
    verify_packed_greedy,
)
from tensorrt_llm._torch.speculative.dspark_planner import (
    SpsCostTable,
    derive_fixed_verifier_budget_candidates,
    select_fixed_verifier_budget,
    select_fixed_verifier_budget_from_traces,
    validate_sps_cost_table_payload,
)
from tensorrt_llm._torch.speculative.dspark_trace import ConfidenceTraceRing


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


def test_dynamic_budget_reducer_selects_global_goodput_not_max_local_tier() -> None:
    budgets = torch.tensor([2, 4, 6], dtype=torch.int64)
    step_times = torch.tensor([1.0, 1.6, 2.4], dtype=torch.float32)
    low_logits = torch.full((2, 2), -20.0, dtype=torch.float32)
    high_logits = torch.full((2, 2), 20.0, dtype=torch.float32)

    def candidate_yields(logits: torch.Tensor) -> torch.Tensor:
        observed = []

        def capture(values: torch.Tensor) -> torch.Tensor:
            observed.append(values.clone())
            return values

        plan_dynamic_verifier_budget(
            logits,
            budgets,
            step_times,
            candidate_yield_reducer=capture,
        )
        return observed[0]

    low_local = plan_dynamic_verifier_budget(low_logits, budgets, step_times)
    high_local = plan_dynamic_verifier_budget(high_logits, budgets, step_times)
    global_yields = candidate_yields(low_logits) + candidate_yields(high_logits)

    def reducer(_local: torch.Tensor) -> torch.Tensor:
        return global_yields

    low_global = plan_dynamic_verifier_budget(
        low_logits,
        budgets,
        step_times,
        candidate_yield_reducer=reducer,
    )
    high_global = plan_dynamic_verifier_budget(
        high_logits,
        budgets,
        step_times,
        candidate_yield_reducer=reducer,
    )

    assert int(low_local.verifier_token_budget) == 2
    assert int(high_local.verifier_token_budget) == 6
    assert int(low_global.verifier_token_budget) == 2
    assert int(high_global.verifier_token_budget) == 2


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


def test_sps_artifact_validation_accepts_exact_measured_tiers() -> None:
    fingerprint = validate_sps_cost_table_payload(
        _sps_payload(),
        verifier_token_budget_tiers={4: [8, 16, 24]},
        max_draft_len=5,
    )

    assert fingerprint["topology"] == "TP8_EP8_attention_DP8"


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
