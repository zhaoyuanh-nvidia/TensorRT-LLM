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
"""Tests for fixed-budget DSpark confidence planning."""

from dataclasses import dataclass

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_confidence import (
    plan_fixed_verifier_budget,
    verify_packed_greedy,
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
