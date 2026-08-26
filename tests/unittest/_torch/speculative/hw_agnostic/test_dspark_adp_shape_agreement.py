# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the cached DSpark attention-DP shape agreement."""

import dataclasses
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import ADPShapeAgreement, CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


class _Batch:
    def __init__(self, generation_requests, *, can_graph=True):
        self.generation_requests = generation_requests
        self.context_requests = []
        self.encoder_requests = []
        self.can_run_cuda_graph = can_graph

    @property
    def batch_size(self):
        return len(self.generation_requests)

    @property
    def num_context_requests(self):
        return 0

    @property
    def num_generation_requests(self):
        return len(self.generation_requests)


def _agreement(batch, *, iteration=9, peer_sizes=(3, 4), reuse=True):
    return ADPShapeAgreement(
        iteration=iteration,
        batch_identity=id(batch),
        local_batch_size=batch.batch_size,
        peer_batch_sizes=peer_sizes,
        all_can_graph=True,
        widest_batch_size=max(peer_sizes),
        graph_batch_size=4,
        draft_len=5,
        padding_ready=True,
        ragged_bucket=16,
        reuse_graph_shape=reuse,
    )


def test_agreement_is_bound_to_iteration_identity_and_padding_phase():
    batch = _Batch([object()] * 3)
    agreement = _agreement(batch)
    assert agreement.matches(batch, 9, padded=False)
    assert not agreement.matches(batch, 8, padded=False)
    assert not agreement.matches(_Batch([object()] * 3), 9, padded=False)

    batch.generation_requests.append(object())
    assert agreement.matches(batch, 9, padded=True)
    assert not agreement.matches(batch, 9, padded=False)


def test_can_queue_reuses_policy_agreement_without_another_collective():
    batch = _Batch([object()] * 3)
    agreement = _agreement(batch)

    def reject_allgather(_value):
        raise AssertionError("cached queue agreement issued another collective")

    runner = SimpleNamespace(
        adp_shape_agreement=agreement,
        adp_shape_debug=False,
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(tp_size=8, tp_allgather=reject_allgather)
    executor.model_engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        cuda_graph_runner=runner,
    )
    executor.kv_connector_manager = None
    executor.drafter = None
    executor.iter_counter = 9
    executor._dspark_dynamic_handled_signature = (
        9,
        id(batch),
        3,
        0,
        3,
    )
    executor._handle_dynamic_draft_len = lambda _batch: None

    assert executor._can_queue(batch) == (True, True)


def test_can_queue_debug_collective_fails_closed_on_disagreement():
    batch = _Batch([object()] * 3)
    agreement = _agreement(batch)
    runner = SimpleNamespace(
        adp_shape_agreement=agreement,
        adp_shape_debug=True,
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(
        tp_size=8,
        tp_allgather=lambda _value: [3, 0],
    )
    executor.model_engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        cuda_graph_runner=runner,
    )
    executor.kv_connector_manager = None
    executor.drafter = None
    executor.iter_counter = 9
    executor._dspark_dynamic_handled_signature = (
        9,
        id(batch),
        3,
        0,
        3,
    )
    executor._handle_dynamic_draft_len = lambda _batch: None

    with pytest.raises(RuntimeError, match="debug collective"):
        executor._can_queue(batch)


def test_padding_reuses_only_a_finalized_ready_agreement():
    requests = [SimpleNamespace(py_verify_len=None, py_verify_cap=None) for _ in range(3)]
    batch = _Batch(requests)
    agreement = _agreement(batch, reuse=True)
    collective_calls = 0

    def shape_allgather(_value):
        nonlocal collective_calls
        collective_calls += 1
        return [[True, 3], [True, 4]]

    dummy = SimpleNamespace(py_verify_len=None, py_verify_cap=None)
    runner = SimpleNamespace(
        enabled=True,
        padding_enabled=True,
        max_supported_batch_size=4,
        supported_batch_sizes=[4],
        dynamic_draft_len_mapping=None,
        enable_encoder_decoder_mixed_cuda_graph=False,
        is_encoder_decoder=False,
        adp_shape_agreement=agreement,
        adp_shape_debug=False,
        padding_dummy_requests={5: dummy},
        spec_config=SimpleNamespace(enable_ragged_verify=False),
        config=SimpleNamespace(
            enable_attention_dp=True,
            mapping=SimpleNamespace(tp_size=8),
            dist=SimpleNamespace(tp_allgather=shape_allgather),
            batch_size=4,
        ),
        _can_run_cuda_graph_batch=lambda candidate: candidate.can_run_cuda_graph,
        _round_up_batch_size_with_draft_len=lambda size, _draft_len: (4 if size <= 4 else 0),
        _get_or_create_padding_dummy=lambda _manager, _draft_len: dummy,
    )

    runner.adp_shape_agreement = dataclasses.replace(agreement, reuse_graph_shape=False)
    assert CUDAGraphRunner._get_padded_batch(runner, batch, SimpleNamespace(), 5) == 1
    assert collective_calls == 1

    batch = _Batch([SimpleNamespace(py_verify_len=None, py_verify_cap=None) for _ in range(3)])
    runner.adp_shape_agreement = _agreement(batch, reuse=True)

    def reject_allgather(_value):
        raise AssertionError("cached padding agreement issued another collective")

    runner.config.dist.tp_allgather = reject_allgather
    assert CUDAGraphRunner._get_padded_batch(runner, batch, SimpleNamespace(), 5) == 1
    assert batch.batch_size == 4
    assert batch.generation_requests[-1] is dummy
