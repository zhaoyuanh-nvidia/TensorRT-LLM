# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph identity and fallback tests for DSpark confidence tiers."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata,
    build_req_idx_per_token,
)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    LlmRequestState,
)
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _assert_dspark_confidence_attention_extent,
    _bind_dspark_confidence_attention_layout,
    _bind_dspark_spec_query_lens,
    _build_dspark_confidence_pack_layout,
    _expand_generation_graph_capture_shapes,
    _refresh_dspark_confidence_graph_generation_lengths,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.dspark import (
    DSparkSpecMetadata,
    DSparkWorker,
    _publish_dspark_confidence_route_outputs,
)
from tensorrt_llm._torch.speculative.dspark_confidence import (
    build_confidence_device_layout,
    validate_confidence_device_layout_row_map,
)
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSampler,
    _clone_dspark_confidence_layout,
)
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig


def request_stub(request_id,
                 effective_len,
                 batch_idx,
                 *,
                 dummy=False,
                 seq_slot=None):
    return SimpleNamespace(
        py_request_id=request_id,
        py_draft_tokens=[11, 12, 13, 14, 15],
        py_draft_tokens_effective_len=effective_len,
        py_batch_idx=batch_idx,
        py_seq_slot=batch_idx if seq_slot is None else seq_slot,
        py_dspark_confidence_route_epoch=None,
        py_dspark_confidence_execution_batch_size=None,
        py_dspark_confidence_verifier_token_budget=None,
        is_dummy=dummy,
        is_cuda_graph_dummy=False,
    )


def runner_stub(config, *, capture=False):
    runner = Mock()
    runner.config = SimpleNamespace(is_draft_model=False)
    runner.max_beam_width = 1
    runner._capture_allowed = capture
    runner.spec_config = config
    runner.confidence_device_layout = None
    runner.confidence_adp_verifier_token_budget = 0
    runner._get_seq_len_mode.return_value = False
    return runner


def _layout_carrier(
    requests,
    retained_lens,
    *,
    execution_g,
    verifier_budget,
    route_epoch,
    engine_generation=17,
):
    retained_lens = torch.tensor(retained_lens, dtype=torch.int32)
    real_count = len(requests)
    real_mask = torch.zeros(execution_g, dtype=torch.bool)
    real_mask[:real_count] = True
    source_slots = torch.zeros(execution_g, dtype=torch.long)
    source_slots[:real_count] = torch.tensor(
        [request.py_seq_slot for request in requests], dtype=torch.long
    )
    layout = build_confidence_device_layout(
        retained_lens,
        real_mask,
        source_slots,
        verifier_budget,
        physical_draft_len=5,
    )
    semantics_host = torch.stack((
        layout.semantic_valid,
        layout.row_map_valid,
        layout.retained_token_count,
        layout.query_token_count,
        layout.cu_query_token_count,
        layout.real_request_count,
        layout.declared_verifier_token_budget,
    )).clone()
    return SimpleNamespace(
        next_draft_lens=retained_lens,
        dspark_confidence_layout=layout,
        dspark_confidence_execution_batch_size=execution_g,
        dspark_confidence_verifier_token_budget=None,
        dspark_confidence_verifier_token_budget_host=(
            layout.verifier_token_budget.clone()),
        dspark_confidence_semantics_host=semantics_host,
        dspark_confidence_query_lens_host=layout.query_lens.clone().cpu(),
        dspark_confidence_budget_ready_event=None,
        dspark_confidence_route_epoch=route_epoch,
        dspark_confidence_physical_draft_len=5,
        dspark_confidence_engine_generation=engine_generation,
        dspark_confidence_request_ids=tuple(
            request.py_request_id for request in requests
        ),
        dspark_confidence_seq_slots=tuple(request.py_seq_slot for request in requests),
    )


_ADP_SEMANTIC_ROUTE_FIELDS = (
    "layout_shapes_ready",
    "base_layout_ready",
    "semantic_exact",
    "execution_g",
    "carried_v",
    "physical_k",
    "route_epoch",
    "real_count",
    "semantic_valid",
    "row_map_valid",
    "retained_count",
    "query_count",
    "cu_query_count",
    "semantic_real_count",
    "declared_v",
)


def _adp_semantic_route(
    execution_g,
    verifier_budget,
    physical_k,
    route_epoch,
    real_count,
    *,
    layout_shapes_ready=True,
    base_layout_ready=True,
    semantic_exact=True,
    semantic_valid=1,
    row_map_valid=1,
    retained_count=None,
    query_count=None,
    cu_query_count=None,
    semantic_real_count=None,
    declared_v=None,
):
    values = {
        "layout_shapes_ready": layout_shapes_ready,
        "base_layout_ready": base_layout_ready,
        "semantic_exact": semantic_exact,
        "execution_g": execution_g,
        "carried_v": verifier_budget,
        "physical_k": physical_k,
        "route_epoch": route_epoch,
        "real_count": real_count,
        "semantic_valid": semantic_valid,
        "row_map_valid": row_map_valid,
        "retained_count": (verifier_budget - execution_g
                           if retained_count is None else retained_count),
        "query_count": (verifier_budget
                        if query_count is None else query_count),
        "cu_query_count": (verifier_budget
                           if cu_query_count is None else cu_query_count),
        "semantic_real_count": (real_count if semantic_real_count is None else
                                semantic_real_count),
        "declared_v": verifier_budget if declared_v is None else declared_v,
    }
    route = [values[field] for field in _ADP_SEMANTIC_ROUTE_FIELDS]
    assert len(route) == len(_ADP_SEMANTIC_ROUTE_FIELDS) == 15
    return route


def _set_adp_semantic_route_field(route, field, value):
    assert len(route) == len(_ADP_SEMANTIC_ROUTE_FIELDS) == 15
    route[_ADP_SEMANTIC_ROUTE_FIELDS.index(field)] = value


def test_adp_semantic_route_fixture_matches_production_schema():
    route = _adp_semantic_route(16, 32, 5, 4, 16)
    assert len(route) == len(_ADP_SEMANTIC_ROUTE_FIELDS) == 15
    assert dict(zip(_ADP_SEMANTIC_ROUTE_FIELDS, route)) == {
        "layout_shapes_ready": True,
        "base_layout_ready": True,
        "semantic_exact": True,
        "execution_g": 16,
        "carried_v": 32,
        "physical_k": 5,
        "route_epoch": 4,
        "real_count": 16,
        "semantic_valid": 1,
        "row_map_valid": 1,
        "retained_count": 16,
        "query_count": 32,
        "cu_query_count": 32,
        "semantic_real_count": 16,
        "declared_v": 32,
    }


def test_selected_full_k_has_explicit_verifier_identity():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers={2: [9, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    config.set_confidence_capture_verifier_token_budget(12)
    runner = runner_stub(config, capture=True)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, None, None, dummy=True),
        request_stub(2, None, None, dummy=True),
    ]
    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
        ),
    )
    assert key.verifier_num_tokens == 12


def test_fixed_capture_inventory_has_compact_and_ordinary_key_per_g():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9, 4: 17},
    )
    enabled, shapes = _expand_generation_graph_capture_shapes([(2, 5), (4, 5)], config)
    assert enabled
    assert set(shapes) == {
        (2, 5, 9),
        (2, 5, None),
        (4, 5, 17),
        (4, 5, None),
    }

    resolved = []
    for graph_batch, _, capture_budget in shapes:
        config.set_confidence_capture_verifier_token_budget(capture_budget)
        resolved.append((graph_batch, config.resolve_confidence_verifier_token_budget(graph_batch)))
    assert set(resolved) == {(2, 9), (2, None), (4, 17), (4, None)}

    config.clear_confidence_capture_verifier_token_budget()
    assert config.resolve_confidence_verifier_token_budget(2) == 9
    assert config.resolve_confidence_verifier_token_budget(4) == 17


def test_native_uniform_inventory_uses_ordinary_dense_k_minus_one_and_k_graphs():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12], 4: [20, 24]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )

    enabled, shapes = _expand_generation_graph_capture_shapes(
        [(2, 5), (4, 5)], config)

    assert enabled
    assert set(shapes) == {
        (2, 4, None),
        (2, 5, None),
        (4, 4, None),
        (4, 5, None),
    }
    assert config.resolve_confidence_native_uniform_draft_len_candidates(
        2) == (4, 5)
    assert config.resolve_confidence_native_uniform_draft_len_candidates(
        4) == (4, 5)


def test_native_uniform_rejects_non_dense_or_more_than_two_tiers():
    with pytest.raises(ValueError, match="dense K-1/K verifier tiers"):
        DSparkDecodingConfig(
            max_draft_len=5,
            speculative_model="/tmp/dummy_model",
            confidence_mode="dynamic_budget",
            confidence_native_uniform=True,
            confidence_verifier_token_budget_tiers={2: [9, 10, 12]},
            confidence_sps_cost_table_path="/tmp/dummy-sps.json",
        )


def test_native_uniform_k4_uses_an_ordinary_dense_graph_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 4, 1),
    ]
    for request in batch.generation_requests:
        request.py_draft_tokens = request.py_draft_tokens[:4]

    key = CUDAGraphRunner.get_graph_key(
        runner_stub(config),
        batch,
        spec_metadata=SimpleNamespace(is_all_greedy_sample=True),
    )

    assert key.draft_len == 4
    assert key.verifier_num_tokens == 0


def test_native_uniform_output_keeps_full_physical_proposals_until_group_agreement():
    outputs = {
        "next_draft_tokens": torch.arange(10, dtype=torch.int32).reshape(2, 5)
    }
    row_lens = torch.full((2,), 4, dtype=torch.int32)
    selected_draft_len = torch.tensor(4, dtype=torch.int32)

    _publish_dspark_confidence_route_outputs(
        outputs,
        row_lens,
        None,
        selected_draft_len,
    )

    assert outputs["next_draft_tokens"].shape == (2, 5)
    assert "next_draft_lens" not in outputs
    assert "next_dspark_confidence_layout" not in outputs
    assert outputs["dspark_confidence_native_uniform"] is True
    assert (
        outputs["dspark_confidence_native_uniform_draft_len"]
        is selected_draft_len
    )
    assert all(value is not None for value in outputs.values())


def test_native_uniform_host_route_truncates_full_proposals_after_group_choice():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    for request in requests:
        request.py_dspark_confidence_execution_batch_size = 2
        request.py_dspark_confidence_route_epoch = 7
    batch = ScheduledRequests()
    batch.generation_requests = requests
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
    )
    executor.disable_overlap_scheduler = True
    executor.enable_attention_dp = False

    selected = PyExecutor._resolve_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 4
    assert [len(request.py_draft_tokens) for request in requests] == [4, 4]
    assert [request.py_draft_tokens_effective_len
            for request in requests] == [4, 4]


def test_native_uniform_adp_route_publishes_one_shot_graph_padding_proof():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    for request in requests:
        request.py_dspark_confidence_execution_batch_size = 2
        request.py_dspark_confidence_route_epoch = 7
    batch = ScheduledRequests()
    batch.generation_requests = requests
    graph_runner = SimpleNamespace(
        confidence_native_uniform_adp_agreed_route=None)
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=graph_runner,
    )
    executor.disable_overlap_scheduler = True
    executor.enable_attention_dp = True
    executor.dist = Mock()
    executor.dist.tp_allgather.side_effect = lambda payload: [payload] * 2

    selected = PyExecutor._resolve_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 4
    assert graph_runner.confidence_native_uniform_adp_agreed_route == (
        2, 7, 4, 2)
    executor.dist.tp_allgather.assert_called_once_with([True, 4, 2, 7])


def test_native_uniform_adp_route_is_finalized_after_full_k_reservation():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    ready_event = Mock()
    ready_event.query.return_value = True
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(4),
        dspark_confidence_request_ids=(1, 2),
        dspark_confidence_seq_slots=(0, 1),
        dspark_confidence_execution_batch_size=2,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    graph_runner = SimpleNamespace(
        confidence_native_uniform_adp_agreed_route=(99, 99, 5, 99),
        _round_up_batch_size=Mock(return_value=2),
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=graph_runner,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = True
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))
    executor.speculation_permanently_disabled = False
    executor.dist = Mock()
    executor.dist.tp_allgather.side_effect = lambda payload: [payload] * 2
    executor._dspark_native_uniform_adp_route_vote = None
    executor._dspark_native_route_counts = None

    can_queue, can_queue_this_rank = PyExecutor._can_queue(executor, batch)
    assert can_queue and can_queue_this_rank
    ready_event.query.assert_not_called()
    executor.dist.tp_allgather.assert_called_once_with(2)

    PyExecutor._handle_dynamic_draft_len(executor, batch)
    reserved_lengths = [len(request.py_draft_tokens) for request in requests]
    assert reserved_lengths == [5, 5]
    assert executor.model_engine.runtime_draft_len == 5
    ready_event.query.assert_not_called()

    selected = PyExecutor._finalize_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 4
    assert [len(request.py_draft_tokens) for request in requests] == [4, 4]
    assert executor.model_engine.runtime_draft_len == 4
    assert graph_runner.confidence_native_uniform_adp_agreed_route == (
        2, 7, 4, 2)
    ready_event.query.assert_called_once_with()
    ready_event.synchronize.assert_not_called()
    assert executor.dist.tp_allgather.call_count == 2
    assert executor.dist.tp_allgather.call_args_list[1].args[0] == [
        True, 4, 2, 7
    ]

    # A connector can recheck the same ScheduledRequests object after route
    # resolution. Preserve the ordinary queue collective without reading or
    # publishing a second route vote for the already-consumed iteration.
    assert PyExecutor._can_queue(executor, batch) == (True, True)
    assert executor.dist.tp_allgather.call_count == 3
    assert executor.dist.tp_allgather.call_args.args[0] == 2
    ready_event.query.assert_called_once_with()
    ready_event.synchronize.assert_not_called()


def test_native_uniform_adp_route_not_ready_falls_back_without_wait():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    ready_event = Mock()
    ready_event.query.return_value = False
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(4),
        dspark_confidence_request_ids=(1, 2),
        dspark_confidence_seq_slots=(0, 1),
        dspark_confidence_execution_batch_size=2,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    graph_runner = SimpleNamespace(
        confidence_native_uniform_adp_agreed_route=(99, 99, 5, 99),
        _round_up_batch_size=Mock(return_value=2),
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=graph_runner,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = True
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))
    executor.speculation_permanently_disabled = False
    executor.dist = Mock()
    executor.dist.tp_allgather.side_effect = lambda payload: [payload] * 2
    executor._dspark_native_uniform_adp_route_vote = None
    executor._dspark_native_route_counts = None

    assert PyExecutor._can_queue(executor, batch) == (True, True)
    PyExecutor._handle_dynamic_draft_len(executor, batch)
    selected = PyExecutor._finalize_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 5
    assert [len(request.py_draft_tokens) for request in requests] == [5, 5]
    assert executor.model_engine.runtime_draft_len == 5
    assert graph_runner.confidence_native_uniform_adp_agreed_route is None
    ready_event.query.assert_called_once_with()
    ready_event.synchronize.assert_not_called()
    assert executor.dist.tp_allgather.call_count == 2
    assert executor.dist.tp_allgather.call_args_list[0].args[0] == 2
    assert executor.dist.tp_allgather.call_args_list[1].args[0] == [
        False, 5, 0, 0
    ]


def test_native_uniform_adp_unconfigured_tail_graph_skips_route_collective():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={128: [640, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    ready_event = Mock()
    ready_event.query.return_value = True
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(4),
        dspark_confidence_execution_batch_size=32,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    graph_runner = SimpleNamespace(
        confidence_native_uniform_adp_agreed_route=(99, 99, 5, 99),
        _round_up_batch_size=Mock(return_value=32),
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=graph_runner,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = True
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))
    executor.speculation_permanently_disabled = False
    executor.dist = Mock()
    executor.dist.tp_allgather.side_effect = lambda payload: [payload] * 2
    executor._dspark_native_uniform_adp_route_vote = None
    executor._dspark_native_uniform_adp_route_eligibility = None
    executor._dspark_native_route_counts = None

    assert PyExecutor._can_queue(executor, batch) == (True, True)
    PyExecutor._handle_dynamic_draft_len(executor, batch)
    selected = PyExecutor._finalize_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 5
    assert executor.model_engine.runtime_draft_len == 5
    assert graph_runner.confidence_native_uniform_adp_agreed_route is None
    ready_event.query.assert_not_called()
    executor.dist.tp_allgather.assert_called_once_with(2)
    assert executor._dspark_native_uniform_adp_route_vote[5] == (
        "unconfigured_graph_bypass")


def test_native_uniform_adp_asymmetric_peers_use_common_max_graph():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={128: [640, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    ready_event = Mock()
    ready_event.query.return_value = True
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(4),
        dspark_confidence_execution_batch_size=128,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    graph_runner = SimpleNamespace(
        confidence_native_uniform_adp_agreed_route=(99, 99, 5, 99),
        _round_up_batch_size=Mock(return_value=128),
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=graph_runner,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = True
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))
    executor.speculation_permanently_disabled = False
    executor.dist = Mock()

    def allgather(payload):
        return [payload] * 2 if isinstance(payload, list) else [2, 100]

    executor.dist.tp_allgather.side_effect = allgather
    executor._dspark_native_uniform_adp_route_vote = None
    executor._dspark_native_uniform_adp_route_eligibility = None
    executor._dspark_native_route_counts = None

    assert PyExecutor._can_queue(executor, batch) == (True, True)
    PyExecutor._handle_dynamic_draft_len(executor, batch)
    selected = PyExecutor._finalize_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 4
    assert graph_runner.confidence_native_uniform_adp_agreed_route == (
        128, 7, 4, 2)
    ready_event.query.assert_called_once_with()
    assert executor.dist.tp_allgather.call_count == 2
    assert executor.dist.tp_allgather.call_args_list[0].args[0] == 2
    assert executor.dist.tp_allgather.call_args_list[1].args[0] == [
        True, 4, 128, 7
    ]


def test_native_uniform_adp_context_peer_bypasses_route_symmetrically():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={128: [640, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    batch.context_requests_chunking = [SimpleNamespace()]
    ready_event = Mock()
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(4),
        dspark_confidence_execution_batch_size=128,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    graph_runner = SimpleNamespace(
        confidence_native_uniform_adp_agreed_route=(99, 99, 5, 99),
        _round_up_batch_size=Mock(return_value=128),
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=graph_runner,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = True
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))
    executor.dist = Mock()
    executor.dist.tp_allgather.return_value = [-3, 2]
    executor._dspark_native_uniform_adp_route_vote = None
    executor._dspark_native_uniform_adp_route_eligibility = None
    executor._dspark_native_route_counts = None

    assert PyExecutor._can_queue(executor, batch) == (True, True)
    selected = PyExecutor._finalize_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 5
    assert graph_runner.confidence_native_uniform_adp_agreed_route is None
    ready_event.query.assert_not_called()
    graph_runner._round_up_batch_size.assert_not_called()
    executor.dist.tp_allgather.assert_called_once_with(-3)
    assert executor._dspark_native_uniform_adp_route_vote[5] == (
        "context_bypass")


def test_static_adp_admission_keeps_plain_scalar_collective():
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 5, 0),
        request_stub(2, 5, 1),
    ]
    graph_runner = SimpleNamespace(
        _round_up_batch_size=Mock(return_value=128))
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=SimpleNamespace(
            is_native_uniform_confidence_enabled=False),
        cuda_graph_runner=graph_runner,
    )
    executor.enable_attention_dp = True
    executor.dist = Mock()
    executor.dist.tp_allgather.return_value = [2, 2]
    executor._dspark_native_uniform_adp_route_eligibility = object()

    assert PyExecutor._can_queue(executor, batch) == (True, True)

    executor.dist.tp_allgather.assert_called_once_with(2)
    graph_runner._round_up_batch_size.assert_not_called()
    assert executor._dspark_native_uniform_adp_route_eligibility is None


@pytest.mark.parametrize("native_uniform", [False, True])
def test_native_uniform_uses_host_vote_instead_of_device_yield_allreduce(
        native_uniform):
    config = SimpleNamespace(
        max_draft_len=5,
        is_confidence_budget_enabled=True,
        is_dynamic_budget_confidence_enabled=True,
        is_native_uniform_confidence_enabled=native_uniform,
    )
    mapping = SimpleNamespace(tp_size=8, enable_attention_dp=True)
    all_reduce = object()
    with patch(
            "tensorrt_llm._torch.speculative.dspark.AllReduce",
            return_value=all_reduce,
    ) as constructor:
        worker = DSparkWorker(config, mapping)

    if native_uniform:
        assert worker._confidence_yield_all_reduce is None
        constructor.assert_not_called()
    else:
        assert worker._confidence_yield_all_reduce is all_reduce
        constructor.assert_called_once()


def test_native_uniform_graph_padding_reuses_adp_route_agreement():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={128: [640, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    runner = _synthetic_capture_runner(config, capture=False)
    runner.confidence_native_uniform_adp_agreed_route = (128, 7, 4, 2)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 4, 1),
    ]
    for request in batch.generation_requests:
        request.py_draft_tokens = request.py_draft_tokens[:4]
    carrier = SimpleNamespace(dspark_confidence_native_uniform=True)

    decision = CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier)

    assert decision == 128
    assert runner.confidence_adp_execution_batch_size == 128
    assert runner.confidence_adp_route_epoch == 7
    assert runner.confidence_force_full_k_route
    assert runner.confidence_native_uniform_adp_agreed_route is None
    runner.config.dist.tp_allgather.assert_not_called()


def test_native_uniform_padding_dummy_is_allocated_lazily():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={128: [640, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    runner = SimpleNamespace(
        enabled=True,
        padding_enabled=True,
        spec_config=config,
        _get_or_create_padding_dummy=Mock(),
    )
    resource_manager = Mock()

    CUDAGraphRunner.preallocate_padding_dummies(runner, resource_manager)

    resource_manager.get_resource_manager.assert_not_called()
    runner._get_or_create_padding_dummy.assert_not_called()


def test_native_uniform_route_trace_flushes_rank_local_counts(tmp_path):
    executor = object.__new__(PyExecutor)
    executor._dspark_native_route_trace_dir = str(tmp_path)
    executor._dspark_native_route_counts = {}
    executor._dspark_native_route_source_counts = {}
    executor.global_rank = 3
    executor.dist = SimpleNamespace(tp_rank=3)

    PyExecutor._record_dspark_native_route(
        executor,
        route_valid=True,
        runtime_draft_len=4,
        execution_batch_size=128,
        real_request_count=127,
        route_source="current",
    )
    PyExecutor._record_dspark_native_route(
        executor,
        route_valid=True,
        runtime_draft_len=4,
        execution_batch_size=128,
        real_request_count=127,
        route_source="current",
    )
    PyExecutor._record_dspark_native_route(
        executor,
        route_valid=False,
        runtime_draft_len=5,
        execution_batch_size=0,
        real_request_count=2,
        route_source="fallback",
    )

    PyExecutor._flush_dspark_native_route_trace(executor)

    payload = json.loads((tmp_path / "rank-3.json").read_text())
    assert payload == {
        "schema": "dspark-native-uniform-route-trace-v1",
        "global_rank": 3,
        "tp_rank": 3,
        "recorded_iterations": 3,
        "routes": [
            {
                "route_valid": False,
                "runtime_draft_len": 5,
                "execution_batch_size": 0,
                "real_request_count": 2,
                "iterations": 1,
            },
            {
                "route_valid": True,
                "runtime_draft_len": 4,
                "execution_batch_size": 128,
                "real_request_count": 127,
                "iterations": 2,
            },
        ],
    }
    source_payload = json.loads(
        (tmp_path / "rank-3-sources.json").read_text())
    assert source_payload == {
        "schema": "dspark-native-uniform-route-source-trace-v1",
        "global_rank": 3,
        "tp_rank": 3,
        "recorded_iterations": 3,
        "sources": [
            {
                "source": "current",
                "iterations": 2,
            },
            {
                "source": "fallback",
                "iterations": 1,
            },
        ],
    }
    assert executor._dspark_native_route_counts is None
    assert executor._dspark_native_route_source_counts is None


def test_native_uniform_asymmetric_host_route_falls_back_to_full_k():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 5, 1)]
    for request in requests:
        request.py_dspark_confidence_execution_batch_size = 2
        request.py_dspark_confidence_route_epoch = 7
    batch = ScheduledRequests()
    batch.generation_requests = requests
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
    )
    executor.disable_overlap_scheduler = True
    executor.enable_attention_dp = False

    selected = PyExecutor._resolve_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 5
    assert [len(request.py_draft_tokens) for request in requests] == [5, 5]
    assert all(request.py_dspark_confidence_route_epoch is None
               for request in requests)


def test_native_uniform_overlap_never_reuses_stale_request_route():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    for request in requests:
        request.py_dspark_confidence_execution_batch_size = 2
        request.py_dspark_confidence_route_epoch = 7
    batch = ScheduledRequests()
    batch.generation_requests = requests
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = False
    executor._dspark_native_uniform_delayed_route = (4, 2, 6, 17)
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(
            device=SimpleNamespace(dspark_confidence_native_uniform=False)))

    selected = PyExecutor._resolve_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == 5
    assert [len(request.py_draft_tokens) for request in requests] == [5, 5]


@pytest.mark.parametrize(("ready", "expected"), [(False, 5), (True, 4)])
def test_native_uniform_overlap_route_never_blocks_for_device_scalar(
        ready, expected):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 5, 0), request_stub(2, 5, 1)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    ready_event = Mock()
    ready_event.query.return_value = ready
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(4),
        dspark_confidence_request_ids=(1, 2),
        dspark_confidence_seq_slots=(0, 1),
        dspark_confidence_execution_batch_size=2,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
    )
    executor.disable_overlap_scheduler = False
    executor.enable_attention_dp = False
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))

    selected = PyExecutor._resolve_dspark_native_uniform_draft_len(
        executor, batch)

    assert selected == expected
    ready_event.query.assert_called_once_with()
    ready_event.synchronize.assert_not_called()
    assert [len(request.py_draft_tokens) for request in requests] == [
        expected, expected
    ]


def test_k6_v816_capture_inventory_and_attention_extents():
    schedule = {16: 102, 32: 204, 64: 408, 128: 816}
    config = DSparkDecodingConfig(
        max_draft_len=6,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule=schedule,
    )

    enabled, shapes = _expand_generation_graph_capture_shapes(
        [(graph_batch_size, 6) for graph_batch_size in schedule], config
    )
    assert enabled
    assert set(shapes) == {
        (graph_batch_size, 6, verifier_token_budget)
        for graph_batch_size, verifier_token_budget in schedule.items()
    } | {(graph_batch_size, 6, None) for graph_batch_size in schedule}

    compact_retained = torch.tensor([6] * 48 + [5] * 80, dtype=torch.int32)
    ordinary_retained = torch.full((128,), 6, dtype=torch.int32)
    observed = []
    for retained, verifier_budget in (
        (compact_retained, 816),
        (ordinary_retained, 896),
        (compact_retained, 816),
    ):
        layout = build_confidence_device_layout(
            retained,
            torch.ones(128, dtype=torch.bool),
            torch.arange(128),
            verifier_budget,
            physical_draft_len=6,
        )
        observed.append(layout.query_lens.clone())
        assert int(layout.query_token_count) == verifier_budget
        assert int(layout.verifier_token_budget) == verifier_budget

    assert observed[0].tolist() == [7] * 48 + [6] * 80
    assert observed[1].tolist() == [7] * 128
    assert torch.equal(observed[0], observed[2])


def test_smaller_g_profile_capture_inventory_has_every_exact_cell():
    tiers = {
        16: [32, 64, 96],
        32: [64, 128, 192],
        64: [128, 256, 384],
        128: [256, 512, 768],
    }
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers=tiers,
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    enabled, shapes = _expand_generation_graph_capture_shapes(
        [(graph_batch_size, 5) for graph_batch_size in tiers], config
    )
    assert enabled
    expected_compact = {
        (graph_batch_size, 5, verifier_token_budget)
        for graph_batch_size, budgets in tiers.items()
        for verifier_token_budget in budgets
    }
    expected_ordinary = {(graph_batch_size, 5, None) for graph_batch_size in tiers}
    assert set(shapes) == expected_compact | expected_ordinary

    resolved = set()
    for graph_batch_size, _, capture_budget in shapes:
        config.set_confidence_capture_verifier_token_budget(capture_budget)
        resolved.add(
            (
                graph_batch_size,
                config.resolve_confidence_verifier_token_budget(graph_batch_size),
            )
        )
    assert resolved == {
        (graph_batch_size, verifier_token_budget)
        for graph_batch_size, _, verifier_token_budget in expected_compact
    } | {(graph_batch_size, None) for graph_batch_size in tiers}


def test_fixed_explicit_ordinary_capture_and_live_fallback_share_v0_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, None, None, dummy=True),
        request_stub(2, None, None, dummy=True),
    ]
    config.set_confidence_capture_verifier_token_budget(None)
    capture_key = CUDAGraphRunner.get_graph_key(
        runner_stub(config, capture=True),
        batch,
        spec_metadata=SimpleNamespace(is_all_greedy_sample=True),
    )
    assert capture_key.verifier_num_tokens == 0
    config.clear_confidence_capture_verifier_token_budget()
    assert config.resolve_confidence_verifier_token_budget(2) == 9


def test_staged_compact_shape_uses_exact_v():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    runner = runner_stub(config)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1),
    ]
    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=SimpleNamespace(),
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
        ),
    )
    assert key.verifier_num_tokens == 9


def test_unstaged_compact_shape_uses_safe_full_k_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    runner = runner_stub(config)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, None),
    ]
    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=SimpleNamespace(),
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
        ),
    )
    assert key.verifier_num_tokens == 0


def test_live_dummy_compact_shape_uses_safe_full_k_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    runner = runner_stub(config)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1, dummy=True),
    ]
    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=SimpleNamespace(),
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
        ),
    )
    assert key is not None
    assert key.verifier_num_tokens == 0


def test_live_dummy_physical_full_k_confidence_uses_safe_full_k_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 12},
    )
    runner = runner_stub(config)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 5, 0),
        request_stub(2, 5, 1, dummy=True),
    ]
    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=SimpleNamespace(),
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
        ),
    )
    assert key is not None
    assert key.verifier_num_tokens == 0


def test_anchor_only_runtime_padding_uses_exact_compact_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={4: 7},
    )
    runner = runner_stub(config)
    first = request_stub(1, 2, 0)
    second = request_stub(2, 1, 1)
    padding = request_stub(999, 0, None, dummy=True)
    padding.is_cuda_graph_dummy = True
    batch = ScheduledRequests()
    batch.generation_requests = [first, second, padding, padding]

    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=SimpleNamespace(),
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=False,
        ),
    )

    assert key is not None
    assert key.batch_size == 4
    assert key.verifier_num_tokens == 7


def test_nonzero_or_non_padding_dummy_cannot_use_compact_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={4: 8},
    )
    runner = runner_stub(config)
    real = [request_stub(1, 2, 0), request_stub(2, 1, 1)]
    non_padding_dummy = request_stub(999, 1, None, dummy=True)
    batch = ScheduledRequests()
    batch.generation_requests = [*real, non_padding_dummy, non_padding_dummy]

    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=SimpleNamespace(),
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=False,
        ),
    )

    assert key is not None
    assert key.verifier_num_tokens == 0


def test_runtime_padding_dummy_gets_anchor_only_effective_length():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers={4: [7, 24]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    runner = Mock()
    runner._capture_allowed = False
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 4
    runner.config = SimpleNamespace(
        is_draft_model=False,
        enable_attention_dp=False,
        mapping=SimpleNamespace(tp_size=1),
        batch_size=4,
    )
    runner.spec_config = config
    runner._can_run_cuda_graph_batch.return_value = True
    runner._round_up_batch_size_with_draft_len.return_value = 4
    padding_dummy = request_stub(999, 5, None, dummy=True)
    padding_dummy.is_cuda_graph_dummy = True
    runner._get_or_create_padding_dummy.return_value = padding_dummy
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 2, 0),
        request_stub(2, 1, 1),
    ]

    assert CUDAGraphRunner._get_padded_batch(runner, batch, Mock(), 5) == 2
    assert padding_dummy.py_draft_tokens_effective_len == 0
    assert batch.generation_requests[-2:] == [padding_dummy, padding_dummy]


def test_native_common_batch_miss_falls_back_to_ordinary_adp_graph_padding():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={4: [20, 24]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    runner = Mock()
    runner._capture_allowed = False
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 4
    runner.spec_config = config
    runner.config = SimpleNamespace(
        is_draft_model=False,
        enable_attention_dp=True,
        batch_size=4,
        dist=Mock(),
    )
    runner.config.dist.tp_allgather.side_effect = [
        [[True, 1, False], [False, 0, False]],
        [[True, 1], [True, 4]],
    ]
    runner._can_run_cuda_graph_batch.return_value = True
    runner._round_up_batch_size_with_draft_len.return_value = 4
    padding_dummy = request_stub(999, 5, None, dummy=True)
    padding_dummy.is_cuda_graph_dummy = True
    runner._get_or_create_padding_dummy.return_value = padding_dummy
    batch = ScheduledRequests()
    batch.generation_requests = [request_stub(1, 5, 0)]

    padding = CUDAGraphRunner._get_padded_batch(runner, batch, Mock(), 5)

    assert padding == 3
    assert runner.confidence_adp_plan_ready is False
    assert batch.generation_requests[-3:] == [padding_dummy] * 3


def test_attention_dp_v_mismatch_fails_before_late_full_k_conversion():
    runner = Mock()
    runner._capture_allowed = False
    runner.enabled = True
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        use_mrope=False,
        mapping=SimpleNamespace(tp_size=8),
        dist=Mock(),
    )
    compact_key = KeyType(
        batch_size=2,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=9,
    )
    full_key = compact_key._replace(verifier_num_tokens=0)
    graph_attn_metadata = object()
    graph_spec_metadata = object()
    runner.get_graph_key.return_value = compact_key
    runner.graph_metadata = {
        full_key: {
            "attn_metadata": graph_attn_metadata,
            "spec_metadata": graph_spec_metadata,
        }
    }
    runner._is_mixed_encoder_decoder_batch.return_value = False
    runner._can_run_cuda_graph_batch.return_value = True
    runner.config.dist.tp_allgather.return_value = [
        [True, 2, 9, 5, True],
        [True, 2, 12, 5, True],
    ]
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1),
    ]
    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.cuda_graph_runner.ExpertStatistic.should_record",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="different verifier shapes"),
    ):
        CUDAGraphRunner.maybe_get_cuda_graph(
            runner,
            batch,
            enable_spec_decode=True,
            attn_metadata=object(),
            new_tensors_device=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "graph_key",
    [
        None,
        KeyType(batch_size=2, draft_len=5, is_first_draft=False),
    ],
)
def test_full_k_fallback_realigns_target_request_state(graph_key):
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    engine.cuda_graph_runner = SimpleNamespace(_capture_allowed=False)
    engine.spec_config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1),
    ]
    padding = request_stub(999, 0, None, dummy=True)
    padding.is_cuda_graph_dummy = True
    batch.generation_requests.append(padding)
    assert engine._align_dspark_confidence_lengths_to_graph(
        batch, graph_key, new_tensors_device=None
    )
    assert [r.py_draft_tokens_effective_len for r in batch.generation_requests] == [
        5,
        5,
        5,
    ]


def test_draft_engine_never_reconstructs_target_verifier_lengths():
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = True
    engine.cuda_graph_runner = SimpleNamespace(_capture_allowed=False)
    engine.spec_config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1),
    ]
    assert not engine._align_dspark_confidence_lengths_to_graph(
        batch, None, new_tensors_device=SimpleNamespace()
    )
    assert [r.py_draft_tokens_effective_len for r in batch.generation_requests] == [4, 3]


def test_compact_graph_preserves_retained_request_lengths():
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    engine.cuda_graph_runner = SimpleNamespace(_capture_allowed=False)
    engine.spec_config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1),
    ]
    compact_key = KeyType(
        batch_size=2,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=9,
    )

    assert not engine._align_dspark_confidence_lengths_to_graph(
        batch, compact_key, new_tensors_device=SimpleNamespace()
    )
    assert [r.py_draft_tokens_effective_len for r in batch.generation_requests] == [4, 3]


def test_synthetic_capture_preserves_capture_owned_dummy_lengths():
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    engine.cuda_graph_runner = SimpleNamespace(_capture_allowed=True)
    engine.spec_config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={2: 9},
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(998, 1, None, dummy=True),
        request_stub(999, 3, None, dummy=True),
    ]
    for request in batch.generation_requests:
        request.is_cuda_graph_dummy = True

    full_key = KeyType(batch_size=2, draft_len=5, is_first_draft=False)
    assert not engine._align_dspark_confidence_lengths_to_graph(
        batch, full_key, new_tensors_device=SimpleNamespace()
    )
    assert [r.py_draft_tokens_effective_len for r in batch.generation_requests] == [1, 3]


def test_late_compact_to_full_k_with_new_tensors_fails_before_cuda():
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    engine.cuda_graph_runner = SimpleNamespace(_capture_allowed=False)
    engine.spec_config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={3: 6},
    )
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 2, 0),
        request_stub(2, 1, 1),
    ]
    padding = request_stub(999, 0, None, dummy=True)
    padding.is_cuda_graph_dummy = True
    batch.generation_requests.append(padding)
    compact_key = KeyType(
        batch_size=3,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=6,
    )
    full_key = compact_key._replace(verifier_num_tokens=0)

    assert not engine._align_dspark_confidence_lengths_to_graph(
        batch, compact_key, new_tensors_device=SimpleNamespace()
    )
    assert sum(1 + r.py_draft_tokens_effective_len for r in batch.generation_requests) == 6

    with pytest.raises(RuntimeError, match="cannot be reconstructed as full-K"):
        engine._align_dspark_confidence_lengths_to_graph(
            batch, full_key, new_tensors_device=SimpleNamespace()
        )
    assert [r.py_draft_tokens_effective_len for r in batch.generation_requests] == [2, 1, 0]


def test_adp_n16_n17_resolves_common_g32_v64_without_oob_indices():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )

    for real_count, retained_lens in (
        (16, [2] * 16),
        (17, [2] * 15 + [1, 1]),
    ):
        runner = Mock()
        runner._capture_allowed = False
        runner.enabled = True
        runner.padding_enabled = True
        runner.max_supported_batch_size = 32
        runner.supported_batch_sizes = [16, 32]
        runner.spec_config = config
        runner._confidence_last_route_epoch = 0
        runner.confidence_engine_generation = 17
        runner.config = SimpleNamespace(
            is_draft_model=False,
            enable_attention_dp=True,
            mapping=SimpleNamespace(tp_size=1),
            dist=Mock(),
            batch_size=32,
        )
        runner.config.dist.tp_allgather.side_effect = lambda payload: [payload] * 2
        runner._can_run_cuda_graph_batch.return_value = False
        runner._round_up_batch_size_with_draft_len.return_value = 32
        runner._get_confidence_adp_common_batch_size = (
            CUDAGraphRunner._get_confidence_adp_common_batch_size.__get__(runner)
        )
        runner._get_or_create_padding_dummy.return_value = request_stub(999, 0, None, dummy=True)
        runner._get_or_create_padding_dummy.return_value.is_cuda_graph_dummy = True
        batch = ScheduledRequests()
        batch.generation_requests = [
            request_stub(index, retained_len, index)
            for index, retained_len in enumerate(retained_lens)
        ]
        carrier = _layout_carrier(
            batch.generation_requests,
            retained_lens + [0] * (32 - real_count),
            execution_g=32,
            verifier_budget=64,
            route_epoch=1,
        )

        padding = CUDAGraphRunner._get_padded_batch(runner, batch, Mock(), 5, carrier)
        assert padding == 32 - real_count
        assert len(batch.generation_requests) == 32
        effective_lens = retained_lens + [0] * padding
        query_lens, input_indices, _, real_tokens = _build_dspark_confidence_pack_layout(
            effective_lens, real_count, 6, 64
        )
        assert len(query_lens) == 32
        assert sum(query_lens) == 64
        assert real_tokens == real_count + sum(retained_lens)
        assert max(input_indices) < real_count * 6
        key_runner = runner_stub(config)
        key_runner.confidence_device_layout = carrier.dspark_confidence_layout
        compact_key = CUDAGraphRunner.get_graph_key(
            key_runner,
            batch,
            new_tensors_device=carrier,
            spec_metadata=SimpleNamespace(
                is_all_greedy_sample=True,
                confidence_fixed_budget_active=True,
                confidence_force_full_k_route=False,
                confidence_verifier_token_budget=64,
            ),
        )
        assert (compact_key.batch_size, compact_key.verifier_num_tokens) == (
            32,
            64,
        )


@pytest.mark.parametrize(
    ("rank_info", "batch_capacity"),
    [
        (
            [[True, 16, False, 0, 0, 0],
             [False, 17, False, 0, 0, 0]],
            32,
        ),
        (
            [[True, 16, False, 0, 0, 0],
             [True, 33, False, 0, 0, 0]],
            32,
        ),
    ],
)
def test_adp_unready_or_over_capacity_uses_uniform_predraft_full_k(rank_info, batch_capacity):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )
    runner = Mock()
    runner._capture_allowed = False
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = batch_capacity
    runner.supported_batch_sizes = [16, 32]
    runner.spec_config = config
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        dist=Mock(),
        batch_size=batch_capacity,
    )
    runner.config.dist.tp_allgather.return_value = rank_info
    batch = ScheduledRequests()
    batch.generation_requests = [request_stub(index, 5, index) for index in range(16)]

    decision = CUDAGraphRunner._get_confidence_adp_common_batch_size(runner, batch, None)
    assert decision == 0


@pytest.mark.parametrize(
    ("real_count", "expected_execution_g"),
    [(1, 16), (17, 32), (33, 64), (65, 128)],
)
def test_partial_confidence_coverage_keeps_smaller_batches_on_ordinary_graphs(
    real_count,
    expected_execution_g,
):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers={128: [256, 512, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    runner = _synthetic_capture_runner(config, capture=False)
    runner.supported_batch_sizes = [16, 32, 64, 128]
    runner._confidence_last_route_epoch = 0
    runner.confidence_engine_generation = 17
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, 5, index) for index in range(real_count)
    ]

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, None
    ) == expected_execution_g
    assert runner.confidence_adp_execution_batch_size == expected_execution_g
    assert runner.confidence_force_full_k_route
    if expected_execution_g < 128:
        assert not config.resolve_confidence_verifier_token_budget_candidates(
            expected_execution_g
        )
    else:
        assert config.resolve_confidence_verifier_token_budget_candidates(128) == (
            256,
            512,
            768,
        )


@pytest.mark.parametrize("peer_count", [1, 8])
def test_adp_actual_group_cardinality_preserves_carrier_route_identity(
    peer_count,
):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )
    runner = Mock()
    runner._capture_allowed = False
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 32
    runner.supported_batch_sizes = [16, 32]
    runner.spec_config = config
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        dist=Mock(),
        batch_size=32,
    )
    runner._confidence_last_route_epoch = 0
    runner.confidence_engine_generation = 17
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, 1, index) for index in range(16)
    ]

    for epoch, execution_g, verifier_budget, effective_len in (
        (1, 16, 32, 1),
        (2, 32, 64, 2),
        (3, 16, 32, 1),
    ):
        for request in batch.generation_requests:
            request.py_draft_tokens_effective_len = 5
        carrier = _layout_carrier(
            batch.generation_requests,
            [effective_len] * 16 + [0] * (execution_g - 16),
            execution_g=execution_g,
            verifier_budget=verifier_budget,
            route_epoch=epoch,
        )
        payload = _adp_semantic_route(
            execution_g, verifier_budget, 5, epoch, 16)
        assert len(payload) == len(_ADP_SEMANTIC_ROUTE_FIELDS) == 15
        runner.config.dist.tp_allgather.return_value = [payload] * peer_count
        assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
            runner, batch, carrier) == execution_g
        assert runner.confidence_adp_execution_batch_size == execution_g
        assert runner.confidence_adp_verifier_token_budget == verifier_budget
        assert runner.confidence_adp_route_epoch == epoch
        assert runner.confidence_engine_generation == (
            carrier.dspark_confidence_engine_generation
        )
        assert runner.confidence_device_layout is carrier.dspark_confidence_layout
        assert (runner.confidence_query_lens_host is
                carrier.dspark_confidence_query_lens_host)
        # The target route and packing are iteration-owned by the accepted
        # carrier. Request fields are published later by SpecSampler after the
        # corresponding host update becomes visible.
        assert all(
            request.py_dspark_confidence_route_epoch is None
            and request.py_dspark_confidence_execution_batch_size is None
            and request.py_dspark_confidence_verifier_token_budget is None
            for request in batch.generation_requests
        )


def test_adp_asymmetric_iteration_carrier_fails_closed_predraft():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )
    runner = Mock()
    runner._capture_allowed = False
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 32
    runner.supported_batch_sizes = [16, 32]
    runner.spec_config = config
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        dist=Mock(),
        batch_size=32,
    )
    runner._confidence_last_route_epoch = 0
    runner.confidence_engine_generation = 17
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, 1, index) for index in range(16)
    ]
    carrier = _layout_carrier(
        batch.generation_requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=4,
    )
    local_route = _adp_semantic_route(16, 32, 5, 4, 16)
    peer_route = _adp_semantic_route(32, 64, 5, 4, 17)
    assert len(local_route) == len(peer_route) == 15
    runner.config.dist.tp_allgather.side_effect = [
        [local_route, peer_route],
        [[False, 16], [False, 17]],
    ]

    with pytest.raises(RuntimeError, match="provenance is missing, stale, or asymmetric"):
        CUDAGraphRunner._get_confidence_adp_common_batch_size(
            runner, batch, carrier)
    assert runner.confidence_force_full_k_route


def test_compact_route_authorization_tracks_final_graph_key():
    metadata = object.__new__(DSparkSpecMetadata)
    metadata.confidence_compact_route_authorized = False
    compact_key = KeyType(
        batch_size=2,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=9,
    )

    assert PyTorchModelEngine._set_dspark_confidence_compact_route_authorization(
        metadata, compact_key
    )
    assert metadata.confidence_compact_route_authorized

    full_key = compact_key._replace(verifier_num_tokens=0)
    assert not PyTorchModelEngine._set_dspark_confidence_compact_route_authorization(
        metadata, full_key
    )
    assert not metadata.confidence_compact_route_authorized

    metadata.confidence_compact_route_authorized = True
    assert not PyTorchModelEngine._set_dspark_confidence_compact_route_authorization(metadata, None)
    assert not metadata.confidence_compact_route_authorized


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ({"next_draft_lens": object()}, True),
        ({
            "dspark_confidence_native_uniform": True,
            "dspark_confidence_native_uniform_draft_len": object(),
        }, True),
        ({"dspark_confidence_native_uniform": True}, False),
        ({
            "dspark_confidence_native_uniform": False,
            "dspark_confidence_native_uniform_draft_len": object(),
        }, False),
        ({}, False),
    ],
)
def test_dspark_confidence_successor_route_detects_native_uniform(outputs,
                                                                  expected):
    assert (PyTorchModelEngine._has_dspark_confidence_successor_route(outputs)
            is expected)


@pytest.mark.parametrize(
    ("execution_g", "synthetic_capture", "expected"),
    [(16, False, True), (0, True, False), (0, False, False)],
)
def test_dspark_confidence_successor_route_seal_skips_synthetic_capture(
        execution_g, synthetic_capture, expected):
    outputs = {
        "dspark_confidence_native_uniform": True,
        "dspark_confidence_native_uniform_draft_len": object(),
    }

    assert PyTorchModelEngine._should_seal_dspark_confidence_successor_route(
        outputs, execution_g, synthetic_capture) is expected


def test_dspark_confidence_successor_route_seal_rejects_packed_route_without_g():
    outputs = {"next_draft_lens": object()}

    with pytest.raises(
            RuntimeError,
            match="without an authorized ADP execution batch"):
        PyTorchModelEngine._should_seal_dspark_confidence_successor_route(
            outputs, 0, False)


@pytest.mark.parametrize("confidence_mode", ["fixed_budget", "dynamic_budget"])
@pytest.mark.parametrize("is_draft_model", [False, True])
def test_adp_general_warmup_without_common_route_disables_compact_plan(
    confidence_mode, is_draft_model
):
    config_kwargs = dict(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode=confidence_mode,
    )
    if confidence_mode == "fixed_budget":
        config_kwargs["confidence_verifier_token_budget_schedule"] = {16: 32}
    else:
        config_kwargs.update(
            confidence_verifier_token_budget_tiers={16: [32, 64]},
            confidence_sps_cost_table_path="/tmp/dummy-sps.json",
        )
    config = DSparkDecodingConfig(**config_kwargs)
    metadata = object.__new__(DSparkSpecMetadata)
    runner = SimpleNamespace(
        config=SimpleNamespace(
            enable_attention_dp=True, is_draft_model=is_draft_model
        ),
        confidence_adp_plan_ready=True,
        confidence_adp_execution_batch_size=0,
        _capture_allowed=False,
    )
    requests = [request_stub(index, 5, index, dummy=True) for index in range(16)]

    assert not PyTorchModelEngine._set_dspark_confidence_adp_plan_readiness(
        metadata, runner, config, True, requests
    )
    assert not metadata.confidence_adp_plan_ready


@pytest.mark.parametrize("route", ["synthetic_capture", "live_common_g"])
def test_adp_explicit_capture_or_live_route_enables_compact_plan(route):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    metadata = object.__new__(DSparkSpecMetadata)
    synthetic_capture = route == "synthetic_capture"
    runner = SimpleNamespace(
        config=SimpleNamespace(enable_attention_dp=True, is_draft_model=False),
        confidence_adp_plan_ready=True,
        confidence_adp_execution_batch_size=0 if synthetic_capture else 16,
        _capture_allowed=synthetic_capture,
    )
    requests = [
        request_stub(index, 5, index, dummy=synthetic_capture)
        for index in range(16)
    ]

    assert PyTorchModelEngine._set_dspark_confidence_adp_plan_readiness(
        metadata, runner, config, synthetic_capture, requests
    )
    assert metadata.confidence_adp_plan_ready


def test_anchor_only_padding_layout_packs_real_rows_and_zero_draft_dummies():
    query_lens, input_indices, draft_indices, real_input_tokens = (
        _build_dspark_confidence_pack_layout(
            retained_lens=[2, 1, 0, 0],
            real_request_count=2,
            runtime_tokens_per_gen_step=6,
            verifier_token_budget=7,
        )
    )

    assert query_lens == [3, 2, 1, 1]
    assert input_indices == [0, 1, 2, 6, 7]
    assert draft_indices == [0, 1, 5]
    assert real_input_tokens == 5


@pytest.mark.parametrize(
    ("retained_lens", "real_request_count", "budget", "match"),
    [
        ([2, 1, 1, 0], 2, 8, "padding rows must retain zero"),
        ([2, 1, 0, 0], 2, 8, "do not match.*budget"),
        ([2, 1, 0, 0], 0, 7, "real_request_count"),
        ([6, 1, 0, 0], 2, 11, "physical draft width"),
    ],
)
def test_anchor_only_padding_layout_rejects_unsafe_shapes(
    retained_lens, real_request_count, budget, match
):
    with pytest.raises(ValueError, match=match):
        _build_dspark_confidence_pack_layout(
            retained_lens=retained_lens,
            real_request_count=real_request_count,
            runtime_tokens_per_gen_step=6,
            verifier_token_budget=budget,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_v1_v2_v1_pack_reuse_clears_visible_tail_and_offsets():
    """A reused maximum-size staging buffer cannot expose a prior V."""
    runtime_width = 6
    real_count = 2
    dense_tokens = torch.arange(4 * runtime_width, device="cuda")
    packed_buffer = torch.full((24,), -1, device="cuda", dtype=torch.long)
    pos_offsets = torch.full((4,), -1, device="cuda", dtype=torch.int32)
    kv_offsets = torch.full((4,), -1, device="cuda", dtype=torch.int32)
    observed = []

    for retained_lens, budget in [([2, 1, 0, 0], 7), ([4, 3, 0, 0], 11), ([2, 1, 0, 0], 7)]:
        query_lens, input_indices, _draft_indices, real_tokens = (
            _build_dspark_confidence_pack_layout(
                retained_lens=retained_lens,
                real_request_count=real_count,
                runtime_tokens_per_gen_step=runtime_width,
                verifier_token_budget=budget,
            )
        )
        visible = packed_buffer[:budget]
        visible.zero_()
        indices = torch.tensor(input_indices, device="cuda", dtype=torch.long)
        visible[:real_tokens].copy_(dense_tokens.index_select(0, indices))
        pos_offsets.zero_()
        kv_offsets.zero_()
        pos_offsets[:real_count].copy_(torch.tensor(query_lens[:real_count], device="cuda"))
        kv_offsets[:real_count].copy_(
            torch.tensor(retained_lens[:real_count], device="cuda", dtype=torch.int32)
        )
        observed.append((visible.clone(), pos_offsets.clone(), kv_offsets.clone()))

    torch.cuda.synchronize()
    assert torch.equal(observed[0][0], observed[2][0])
    assert observed[1][0].numel() == 11
    assert torch.count_nonzero(observed[1][0][9:]) == 0
    for _visible, pos, kv in observed:
        assert torch.count_nonzero(pos[real_count:]) == 0
        assert torch.count_nonzero(kv[real_count:]) == 0


def test_confidence_graph_pools_are_v_isolated():
    key = KeyType(batch_size=128, draft_len=5, is_first_draft=False)
    assert CUDAGraphRunner._get_memory_pool_key(key) == (128, 5, 0)
    assert CUDAGraphRunner._get_memory_pool_key(key._replace(verifier_num_tokens=256)) == (
        128,
        5,
        256,
    )
    assert CUDAGraphRunner._get_memory_pool_key(key._replace(verifier_num_tokens=512)) == (
        128,
        5,
        512,
    )


class _MockGraphAttentionMetadata:
    """Model-facing metadata stub whose consumers read ``num_tokens``."""

    def __init__(self):
        self._seq_lens = None
        self.num_tokens = 0

    @property
    def seq_lens(self):
        return self._seq_lens

    @seq_lens.setter
    def seq_lens(self, value):
        self._seq_lens = value
        self.num_tokens = int(value.sum().item())


@pytest.mark.parametrize("execution_batch_size", [16, 32, 64, 128])
@pytest.mark.parametrize("ratio", [2, 4, 6])
def test_compact_graph_capture_metadata_has_exact_token_extent(
    execution_batch_size, ratio
):
    """All 12 profiled (G,V) cells expose V to inputs and model metadata."""
    verifier_num_tokens = execution_batch_size * ratio
    key = KeyType(
        batch_size=execution_batch_size,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=verifier_num_tokens,
    )
    graph_metadata = _MockGraphAttentionMetadata()

    CUDAGraphRunner._initialize_generation_graph_metadata_extent(
        graph_metadata, key, max_beam_width=1
    )

    assert CUDAGraphRunner._get_num_tokens_for_key(None, key) == verifier_num_tokens
    assert len(graph_metadata.seq_lens) == execution_batch_size
    assert graph_metadata.num_tokens == verifier_num_tokens
    assert graph_metadata.seq_lens.min().item() >= 1
    assert graph_metadata.seq_lens.max().item() <= 6


def test_graph_capture_metadata_v1_v2_v1_overwrites_extent_and_ordinary_is_full_k():
    graph_metadata = _MockGraphAttentionMetadata()
    compact_v1 = KeyType(
        batch_size=128,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=256,
    )
    compact_v2 = compact_v1._replace(verifier_num_tokens=512)
    ordinary = compact_v1._replace(verifier_num_tokens=0)
    observed = []

    for key in (compact_v1, compact_v2, compact_v1, ordinary):
        CUDAGraphRunner._initialize_generation_graph_metadata_extent(
            graph_metadata, key, max_beam_width=1
        )
        observed.append(
            (graph_metadata.seq_lens.tolist(), graph_metadata.num_tokens)
        )

    assert [extent for _seq_lens, extent in observed] == [256, 512, 256, 768]
    assert observed[0] == observed[2]
    assert observed[3][0] == [6] * 128


def _synthetic_capture_runner(config, *, capture=True):
    runner = Mock()
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 128
    runner.supported_batch_sizes = [128]
    runner.max_beam_width = 1
    runner.spec_config = config
    runner._capture_allowed = capture
    runner._get_seq_len_mode.return_value = False
    runner._get_num_tokens_for_key = (
        CUDAGraphRunner._get_num_tokens_for_key.__get__(runner)
    )
    runner.config = SimpleNamespace(
        is_draft_model=False,
        enable_attention_dp=True,
        use_mrope=False,
        batch_size=128,
        dist=Mock(),
    )
    runner.config.dist.tp_allgather.side_effect = lambda payload: [payload] * 8
    runner.shared_static_tensors = {
        "input_ids": torch.empty(768, dtype=torch.int32),
        "position_ids": torch.empty((1, 768), dtype=torch.int32),
    }
    return runner


@pytest.mark.parametrize(
    ("capture_budget", "expected_tokens", "expected_compact"),
    [(256, 256, True), (None, 768, False)],
)
def test_synthetic_capture_route_keeps_model_facing_extent_consistent(
    capture_budget, expected_tokens, expected_compact
):
    """Exercise startup ordering from ADP consensus through model inputs."""
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers={128: [256, 512, 768]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    config.set_confidence_capture_verifier_token_budget(capture_budget)
    runner = _synthetic_capture_runner(config)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, None, None, dummy=True) for index in range(128)
    ]

    common_g = CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, None
    )
    assert common_g == 128
    assert runner.confidence_adp_execution_batch_size == 128
    assert runner.confidence_force_full_k_route is not expected_compact

    spec_metadata = SimpleNamespace(
        is_all_greedy_sample=True,
        confidence_fixed_budget_active=expected_compact,
        confidence_force_full_k_route=runner.confidence_force_full_k_route,
    )
    key = CUDAGraphRunner.get_graph_key(
        runner, batch, spec_metadata=spec_metadata
    )
    assert key.verifier_num_tokens == (256 if expected_compact else 0)
    assert CUDAGraphRunner._get_num_tokens_for_key(runner, key) == expected_tokens

    graph_metadata = _MockGraphAttentionMetadata()
    CUDAGraphRunner._initialize_generation_graph_metadata_extent(
        graph_metadata, key, max_beam_width=1
    )
    static_inputs = CUDAGraphRunner._get_capture_static_model_inputs(
        runner, key, {"attn_metadata": graph_metadata}
    )
    model_inputs = {**static_inputs, "attn_metadata": graph_metadata}
    CUDAGraphRunner._assert_generation_model_token_extent(key, model_inputs)

    assert static_inputs["input_ids"].shape[0] == expected_tokens
    assert static_inputs["position_ids"].shape[-1] == expected_tokens
    assert graph_metadata.num_tokens == expected_tokens
    assert PyTorchModelEngine._is_dspark_confidence_input_packing_allowed(
        spec_metadata
    ) is expected_compact


def test_synthetic_capture_peer_shape_disagreement_fails_closed():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers={128: [256, 512]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    config.set_confidence_capture_verifier_token_budget(256)
    runner = _synthetic_capture_runner(config)
    runner.config.dist.tp_allgather.side_effect = lambda payload: [
        payload,
        [True, 128, 5, True, 512],
    ]
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, None, None, dummy=True) for index in range(128)
    ]

    with pytest.raises(RuntimeError, match="different synthetic capture shapes"):
        CUDAGraphRunner._get_confidence_adp_common_batch_size(runner, batch, None)
    assert runner.confidence_force_full_k_route


def test_synthetic_capture_malformed_participant_fails_closed():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={128: 256},
    )
    config.set_confidence_capture_verifier_token_budget(256)
    runner = _synthetic_capture_runner(config)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request_stub(900, None, None)]

    with pytest.raises(RuntimeError, match="requires an all-dummy generation batch"):
        CUDAGraphRunner._get_confidence_adp_common_batch_size(runner, batch, None)
    assert runner.config.dist.tp_allgather.call_args.args[0] == [
        False,
        0,
        -1,
        True,
        256,
    ]
    assert runner.confidence_force_full_k_route


def test_synthetic_capture_rejects_heterogeneous_collective_payload_schema():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={128: 256},
    )
    config.set_confidence_capture_verifier_token_budget(256)
    runner = _synthetic_capture_runner(config)
    runner.config.dist.tp_allgather.side_effect = lambda payload: [
        payload,
        [False, 0, False],
    ]
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, None, None, dummy=True) for index in range(128)
    ]

    with pytest.raises(RuntimeError, match="malformed peer payloads"):
        CUDAGraphRunner._get_confidence_adp_common_batch_size(runner, batch, None)
    assert runner.confidence_force_full_k_route


def test_non_capture_all_dummy_batch_fails_closed_to_full_k():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={128: 256},
    )
    config.set_confidence_capture_verifier_token_budget(256)
    runner = _synthetic_capture_runner(config, capture=False)
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(index, None, None, dummy=True) for index in range(128)
    ]

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, None
    ) == 0
    assert runner.config.dist.tp_allgather.call_args.args[0] == [False, 0, False]
    assert runner.confidence_force_full_k_route


@pytest.mark.parametrize("local_role", ["rank0_context", "peer_runtime_dummy"])
def test_post_capture_live_context_and_runtime_dummy_peers_share_ordinary_schema(
    local_role,
):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={128: 256},
    )
    config.clear_confidence_capture_verifier_token_budget()
    runner = _synthetic_capture_runner(config, capture=False)
    exchanged_payloads = []

    def ordinary_consensus(payload):
        exchanged_payloads.append(payload)
        return [[False, 0, False]] * 8

    runner.config.dist.tp_allgather.side_effect = ordinary_consensus
    batch = ScheduledRequests()
    if local_role == "rank0_context":
        batch.context_requests_last_chunk = [request_stub(1024, None, None)]
    else:
        batch.generation_requests = [
            request_stub(
                ATTENTION_DP_DUMMY_REQUEST_ID,
                None,
                None,
                dummy=True,
            )
        ]

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, None
    ) == 0
    assert exchanged_payloads == [[False, 0, False]]
    assert runner.confidence_force_full_k_route


def test_model_facing_extent_guard_catches_hyperconnection_counterexample():
    key = KeyType(
        batch_size=128,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=0,
    )
    model_inputs = {
        "input_ids": torch.empty(768, dtype=torch.int32),
        "position_ids": torch.empty((1, 768), dtype=torch.int32),
        "attn_metadata": SimpleNamespace(num_tokens=256),
    }

    with pytest.raises(RuntimeError, match="model-facing token extents"):
        CUDAGraphRunner._assert_generation_model_token_extent(key, model_inputs)


@pytest.mark.parametrize("verifier_num_tokens", [127, 769])
def test_compact_graph_capture_metadata_rejects_out_of_physical_extent(
    verifier_num_tokens,
):
    key = KeyType(
        batch_size=128,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=verifier_num_tokens,
    )
    with pytest.raises(ValueError, match="physical"):
        CUDAGraphRunner._get_generation_seq_lens_for_key(key, max_beam_width=1)


def _make_sampler_route_state(*, execution_g=4, route_epoch=7):
    requests = [
        SimpleNamespace(
            state=LlmRequestState.GENERATION_IN_PROGRESS,
            py_seq_slot=slot,
            py_request_id=100 + slot,
            py_return_context_logits=False,
            py_return_generation_logits=False,
            py_return_log_probs=False,
            py_draft_tokens=[10, 11, 12, 13, 14],
            py_draft_tokens_effective_len=5,
            py_decoding_iter=0,
            py_dspark_confidence_route_epoch=None,
            py_dspark_confidence_execution_batch_size=None,
            py_dspark_confidence_verifier_token_budget=None,
        )
        for slot in range(2)
    ]
    state = object.__new__(SampleStateSpec)
    state.sampler_event = SimpleNamespace(synchronize=lambda: None)
    state.requests = requests
    state.draft_lens = [5, 5]
    state.runtime_draft_len = 5
    state.dspark_confidence_execution_batch_size = execution_g
    state.dspark_confidence_route_epoch = route_epoch
    state.dspark_confidence_verifier_token_budget = None
    state.dspark_confidence_physical_draft_len = 5
    state.dspark_confidence_engine_generation = 17
    state.host = SimpleNamespace(
        new_tokens=torch.zeros((2, 6), dtype=torch.int32),
        new_tokens_lens=torch.ones(2, dtype=torch.int32),
        next_draft_tokens=torch.tensor(
            [[20, 21, 22, 23, 24], [30, 31, 32, 33, 34]], dtype=torch.int32
        ),
        next_draft_lens=torch.tensor([1, 2], dtype=torch.int32),
        verified_draft_lens=torch.tensor([5, 5], dtype=torch.int32),
        dspark_confidence_verifier_token_budget_host=torch.tensor(
            execution_g + 3, dtype=torch.int32),
        dspark_confidence_semantics_host=torch.tensor(
            [1, 1, 3, execution_g + 3, execution_g + 3, 2, execution_g + 3],
            dtype=torch.int32),
    )
    sampler = object.__new__(SpecSampler)
    sampler.max_accepted_path_len = 6
    sampler.max_seq_len = 4096
    sampler.draft_len = 5
    sampler._trace_dspark_budget = False
    return sampler, state, requests


def test_sampler_publishes_iteration_local_compact_route(monkeypatch):
    sampler, state, requests = _make_sampler_route_state()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    SpecSampler.update_requests(sampler, state)

    assert [request.py_draft_tokens_effective_len for request in requests] == [1, 2]
    assert all(len(request.py_draft_tokens) == 5 for request in requests)
    assert all(request.py_dspark_confidence_route_epoch == 7 for request in requests)
    assert all(
        request.py_dspark_confidence_execution_batch_size == 4
        for request in requests
    )
    assert all(
        request.py_dspark_confidence_verifier_token_budget == 7
        for request in requests
    )


def test_sampler_carries_native_uniform_route_after_natural_wait(monkeypatch):
    sampler, state, requests = _make_sampler_route_state()
    state.host.next_draft_lens = None
    state.host.dspark_confidence_native_uniform_draft_len_host = torch.tensor(
        4, dtype=torch.int32)
    state.dspark_confidence_native_uniform = True
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    SpecSampler.update_requests(sampler, state)

    assert [request.py_draft_tokens_effective_len for request in requests] == [4, 4]
    assert all(len(request.py_draft_tokens) == 5 for request in requests)
    assert all(request.py_dspark_confidence_route_epoch == 7 for request in requests)
    assert all(
        request.py_dspark_confidence_execution_batch_size == 4
        for request in requests
    )
    assert all(
        request.py_dspark_confidence_verifier_token_budget == 20
        for request in requests
    )


def test_native_uniform_executor_delayed_route_survives_roster_churn():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(1, 4, 0), request_stub(2, 4, 1)]
    # Request-local metadata may be missing or heterogeneous when requests
    # finish and are admitted between the completed and queued iterations.
    requests[0].py_dspark_confidence_execution_batch_size = 4
    requests[0].py_dspark_confidence_route_epoch = 6
    requests[1].py_draft_tokens_effective_len = 5
    batch = ScheduledRequests()
    batch.generation_requests = requests
    ready_event = Mock()
    ready_event.query.return_value = False
    previous_device = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_native_uniform_ready_event=ready_event,
        dspark_confidence_native_uniform_draft_len_host=torch.tensor(5),
        # This carrier was planned before one request completed and another
        # occupied its slot. Uniform prefix truncation has no row mapping, so
        # the graph runner's physical-block validation is the safety proof.
        dspark_confidence_request_ids=(10, 20),
        dspark_confidence_seq_slots=(5, 6),
        dspark_confidence_execution_batch_size=2,
        dspark_confidence_route_epoch=7,
        dspark_confidence_engine_generation=17,
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
        cuda_graph_runner=SimpleNamespace(
            confidence_native_uniform_adp_agreed_route=None),
    )
    executor.disable_overlap_scheduler = False
    executor._dspark_native_uniform_delayed_route = (4, 4, 6, 17)
    executor.previous_batch = SimpleNamespace(
        sample_state=SimpleNamespace(device=previous_device))

    assert PyExecutor._get_dspark_native_uniform_local_route(
        executor, batch) == (True, 4, 2, 7)
    ready_event.query.assert_called_once_with()


def test_executor_publishes_native_uniform_route_after_sampler_wait():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_native_uniform=True,
        confidence_verifier_token_budget_tiers={2: [10, 12]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=5,
        _dspark_confidence_engine_generation=17,
    )
    executor._dspark_native_uniform_delayed_route = None
    state = SimpleNamespace(
        dspark_confidence_native_uniform=True,
        dspark_confidence_execution_batch_size=2,
        dspark_confidence_route_epoch=6,
        dspark_confidence_engine_generation=17,
        host=SimpleNamespace(
            dspark_confidence_native_uniform_draft_len_host=torch.tensor(4)),
    )

    PyExecutor._publish_dspark_native_uniform_delayed_route(executor, state)

    assert executor._dspark_native_uniform_delayed_route == (4, 2, 6, 17)

    state.dspark_confidence_engine_generation = 18
    PyExecutor._publish_dspark_native_uniform_delayed_route(executor, state)
    assert executor._dspark_native_uniform_delayed_route is None


def test_sampler_newer_route_publication_is_monotonic(monkeypatch):
    sampler, state, requests = _make_sampler_route_state(route_epoch=7)
    for request in requests:
        request.py_dspark_confidence_route_epoch = 6
        request.py_dspark_confidence_execution_batch_size = 2
        request.py_dspark_confidence_verifier_token_budget = 4
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    SpecSampler.update_requests(sampler, state)

    assert all(request.py_dspark_confidence_route_epoch == 7 for request in requests)
    assert all(
        request.py_dspark_confidence_execution_batch_size == 4
        for request in requests
    )
    assert all(
        request.py_dspark_confidence_verifier_token_budget == 7
        for request in requests
    )


def test_sampler_equal_route_publication_is_idempotent(monkeypatch):
    sampler, state, requests = _make_sampler_route_state(route_epoch=7)
    for request in requests:
        request.py_dspark_confidence_route_epoch = 7
        request.py_dspark_confidence_execution_batch_size = 4
        request.py_dspark_confidence_verifier_token_budget = 7
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    SpecSampler.update_requests(sampler, state)

    assert all(request.py_dspark_confidence_route_epoch == 7 for request in requests)
    assert all(
        request.py_dspark_confidence_execution_batch_size == 4
        for request in requests
    )
    assert all(
        request.py_dspark_confidence_verifier_token_budget == 7
        for request in requests
    )


def test_sampler_older_route_cannot_overwrite_newer_publication(monkeypatch):
    sampler, state, requests = _make_sampler_route_state(route_epoch=7)
    for request in requests:
        request.py_dspark_confidence_route_epoch = 8
        request.py_dspark_confidence_execution_batch_size = 8
        request.py_dspark_confidence_verifier_token_budget = 12
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    with pytest.raises(RuntimeError, match="route epoch regressed"):
        SpecSampler.update_requests(sampler, state)

    assert all(request.py_dspark_confidence_route_epoch == 8 for request in requests)
    assert all(
        request.py_dspark_confidence_execution_batch_size == 8
        for request in requests
    )
    assert all(
        request.py_dspark_confidence_verifier_token_budget == 12
        for request in requests
    )


def test_sampler_matching_epoch_full_k_route_clears_publication(monkeypatch):
    sampler, state, requests = _make_sampler_route_state(route_epoch=7)
    state.dspark_confidence_execution_batch_size = 0
    state.dspark_confidence_verifier_token_budget = 0
    for request in requests:
        request.py_dspark_confidence_route_epoch = 7
        request.py_dspark_confidence_execution_batch_size = 4
        request.py_dspark_confidence_verifier_token_budget = 7
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    SpecSampler.update_requests(sampler, state)

    assert all(request.py_draft_tokens_effective_len == 5 for request in requests)
    assert all(
        request.py_dspark_confidence_route_epoch is None
        and request.py_dspark_confidence_execution_batch_size is None
        and request.py_dspark_confidence_verifier_token_budget is None
        for request in requests
    )


def test_sampler_invalid_device_semantics_restores_physical_k(monkeypatch):
    sampler, state, requests = _make_sampler_route_state(route_epoch=7)
    state.host.next_draft_lens = torch.tensor([5, 0], dtype=torch.int32)
    state.host.dspark_confidence_verifier_token_budget_host = torch.tensor(
        0, dtype=torch.int32)
    state.host.dspark_confidence_semantics_host = torch.tensor(
        [0, 0, 5, 9, 9, 1, 7], dtype=torch.int32)
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        TorchSampler, "_handle_stop_criteria", lambda *args, **kwargs: False
    )

    SpecSampler.update_requests(sampler, state)

    assert all(request.py_draft_tokens_effective_len == 5
               for request in requests)
    assert all(
        request.py_dspark_confidence_route_epoch is None
        and request.py_dspark_confidence_execution_batch_size is None
        and request.py_dspark_confidence_verifier_token_budget is None
        for request in requests)


def test_sampler_completion_invalidates_compact_route(monkeypatch):
    sampler, state, requests = _make_sampler_route_state()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.spec_sampler_base.add_token",
        lambda *args, **kwargs: 1,
    )

    def complete_first_request(request, *args, **kwargs):
        if request is requests[0]:
            request.state = LlmRequestState.GENERATION_COMPLETE
            return True
        return False

    monkeypatch.setattr(TorchSampler, "_handle_stop_criteria", complete_first_request)

    SpecSampler.update_requests(sampler, state)

    assert requests[0].state == LlmRequestState.GENERATION_COMPLETE
    assert requests[1].py_draft_tokens_effective_len == 5
    assert len(requests[1].py_draft_tokens) == 5
    assert requests[1].py_dspark_confidence_route_epoch is None
    assert requests[1].py_dspark_confidence_execution_batch_size is None
    assert requests[1].py_dspark_confidence_verifier_token_budget is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA metadata storage")
def test_real_attention_metadata_reuses_device_buffer_v1_v2_v1():
    graph_metadata = AttentionMetadata(
        max_num_requests=128,
        max_num_tokens=768,
        is_cuda_graph=True,
    )
    graph_metadata.seq_lens = torch.full((128,), 6, dtype=torch.int32)
    torch.cuda.synchronize()
    stable_device_ptr = graph_metadata.seq_lens_cuda.data_ptr()
    compact_v1 = KeyType(
        batch_size=128,
        draft_len=5,
        is_first_draft=False,
        verifier_num_tokens=256,
    )
    compact_v2 = compact_v1._replace(verifier_num_tokens=512)
    observed = []

    for key in (compact_v1, compact_v2, compact_v1):
        CUDAGraphRunner._initialize_generation_graph_metadata_extent(
            graph_metadata, key, max_beam_width=1
        )
        torch.cuda.synchronize()
        observed.append(graph_metadata.seq_lens_cuda.cpu().tolist())
        assert graph_metadata.seq_lens_cuda.data_ptr() == stable_device_ptr
        assert graph_metadata.seq_lens_cuda.tolist() == graph_metadata.seq_lens.tolist()

    assert sum(observed[0]) == 256
    assert sum(observed[1]) == 512
    assert observed[0] == observed[2]


def _cpu_attention_metadata_with_physical_g128_k5():
    metadata = object.__new__(AttentionMetadata)
    metadata.is_cuda_graph = True
    metadata._seq_lens = torch.full((128,), 6, dtype=torch.int32)
    metadata._seq_lens_kv = None
    metadata._seq_lens_cuda = torch.empty(128, dtype=torch.int32)
    metadata._num_contexts = 0
    metadata._num_ctx_tokens = 0
    metadata._num_generations = 0
    metadata._num_tokens = 0
    metadata.confidence_fixed_budget_active = True
    metadata.spec_decoding_generation_lengths = torch.empty(
        128, dtype=torch.int32)
    metadata.spec_decoding_position_offsets = torch.arange(
        6, dtype=torch.int32).repeat(128)
    metadata.on_update()
    return metadata


def _g128_k5_ragged_retained_lens(verifier_budget, device=None):
    if verifier_budget == 256:
        values = [5] * 16 + [3] * 16 + [0] * 96
    elif verifier_budget == 512:
        values = [5] * 64 + [1] * 64
    elif verifier_budget == 768:
        values = [5] * 128
    else:
        raise ValueError(verifier_budget)
    retained = torch.tensor(values, dtype=torch.int32, device=device)
    assert 128 + int(retained.cpu().sum()) == verifier_budget
    return retained


@pytest.mark.parametrize("verifier_budget", [256, 512, 768])
def test_g128_k5_exact_attention_bind_follows_all_on_update_setters(
    verifier_budget,
):
    retained = _g128_k5_ragged_retained_lens(verifier_budget)
    layout = build_confidence_device_layout(
        retained,
        torch.ones(128, dtype=torch.bool),
        torch.arange(128),
        verifier_budget,
        physical_draft_len=5,
        packed_capacity=768,
    )
    metadata = _cpu_attention_metadata_with_physical_g128_k5()
    stable_host = metadata.seq_lens
    stable_host_ptr = stable_host.data_ptr()
    stable_device_ptr = metadata.seq_lens_cuda.data_ptr()

    assert metadata.num_tokens == 768
    metadata.num_contexts = 0
    assert metadata.num_tokens == 768
    exact_host_query_lens = layout.query_lens.clone().cpu()
    _bind_dspark_confidence_attention_layout(
        metadata, layout, exact_host_query_lens, 128, verifier_budget)

    _assert_dspark_confidence_attention_extent(
        metadata, verifier_budget, "CPU DSA prepare entry")
    assert metadata.num_tokens == verifier_budget
    assert metadata.seq_lens.tolist() == exact_host_query_lens.tolist()
    assert int(metadata._seq_lens_cuda.sum()) == verifier_budget
    assert metadata.seq_lens is stable_host
    assert metadata.seq_lens.data_ptr() == stable_host_ptr
    assert metadata.seq_lens_cuda.data_ptr() == stable_device_ptr


def test_g128_k5_exact_attention_bind_v1_v2_v1_preserves_graph_storage():
    metadata = _cpu_attention_metadata_with_physical_g128_k5()
    stable_host = metadata.seq_lens
    stable_host_ptr = stable_host.data_ptr()
    stable_device_ptr = metadata.seq_lens_cuda.data_ptr()
    stable_generation_ptr = metadata.spec_decoding_generation_lengths.data_ptr()
    stable_position_ptr = metadata.spec_decoding_position_offsets.data_ptr()
    observed = []

    for verifier_budget in (256, 512, 768, 256):
        retained = _g128_k5_ragged_retained_lens(verifier_budget)
        layout = build_confidence_device_layout(
            retained,
            torch.ones(128, dtype=torch.bool),
            torch.arange(128),
            verifier_budget,
            physical_draft_len=5,
            packed_capacity=768,
        )
        event_owned_host = layout.query_lens.clone().cpu()
        assert event_owned_host.data_ptr() != stable_host_ptr
        _bind_dspark_confidence_attention_layout(
            metadata, layout, event_owned_host, 128, verifier_budget)
        observed.append(metadata.seq_lens.clone())
        assert metadata.seq_lens is stable_host
        assert metadata.seq_lens.data_ptr() == stable_host_ptr
        assert metadata.seq_lens_cuda.data_ptr() == stable_device_ptr
        assert (metadata.spec_decoding_generation_lengths.data_ptr()
                == stable_generation_ptr)
        assert metadata.spec_decoding_position_offsets.data_ptr() == stable_position_ptr
        assert torch.equal(
            metadata.spec_decoding_generation_lengths[:128],
            layout.query_lens)
        assert metadata.num_tokens == verifier_budget

    assert [int(lengths.sum()) for lengths in observed] == [256, 512, 768, 256]
    assert torch.equal(observed[0], observed[-1])


def test_g128_v256_ragged_graph_generation_lengths_replace_capture_balance():
    metadata = _cpu_attention_metadata_with_physical_g128_k5()
    metadata.spec_decoding_generation_lengths.fill_(2)
    stable_generation_ptr = metadata.spec_decoding_generation_lengths.data_ptr()
    stable_position_ptr = metadata.spec_decoding_position_offsets.data_ptr()
    query_lens = _g128_k5_ragged_retained_lens(256) + 1

    assert int(query_lens.sum()) == 256
    assert query_lens.min() == 1
    assert query_lens.max() == 6
    assert not torch.equal(metadata.spec_decoding_generation_lengths,
                           query_lens)
    _refresh_dspark_confidence_graph_generation_lengths(
        metadata, query_lens, 128)

    assert torch.equal(metadata.spec_decoding_generation_lengths, query_lens)
    assert (metadata.spec_decoding_generation_lengths.data_ptr()
            == stable_generation_ptr)
    assert metadata.spec_decoding_position_offsets.data_ptr() == stable_position_ptr


def test_shared_cross_tier_generation_buffer_is_refreshed_before_v256_replay():
    base_metadata = _cpu_attention_metadata_with_physical_g128_k5()
    graph_metadata = {
        verifier_budget: copy.copy(base_metadata)
        for verifier_budget in (256, 512, 768)
    }
    shared_generation_lengths = base_metadata.spec_decoding_generation_lengths
    stable_generation_ptr = shared_generation_lengths.data_ptr()
    stable_position_ptr = base_metadata.spec_decoding_position_offsets.data_ptr()
    for metadata in graph_metadata.values():
        assert metadata.spec_decoding_generation_lengths is shared_generation_lengths
        assert metadata.spec_decoding_position_offsets is base_metadata.spec_decoding_position_offsets

    # create_cuda_graph_metadata shallow-copies this one base tensor into every
    # compact graph key. Sequential V256/V512/V768 capture leaves the last
    # tier's balanced boundaries visible to all three cached graphs.
    for capture_budget in (256, 512, 768):
        shared_generation_lengths.fill_(capture_budget // 128)
        assert shared_generation_lengths.data_ptr() == stable_generation_ptr
    assert int(shared_generation_lengths.sum()) == 768

    observed = []
    for replay_budget in (256, 512, 768, 256):
        live_query_lens = (
            _g128_k5_ragged_retained_lens(replay_budget) + 1)
        if replay_budget == 256 and not observed:
            assert not torch.equal(shared_generation_lengths,
                                   live_query_lens)
        _refresh_dspark_confidence_graph_generation_lengths(
            graph_metadata[replay_budget], live_query_lens, 128)

        assert torch.equal(shared_generation_lengths, live_query_lens)
        assert int(shared_generation_lengths.sum()) == replay_budget
        assert shared_generation_lengths.data_ptr() == stable_generation_ptr
        assert (graph_metadata[replay_budget].spec_decoding_position_offsets.
                data_ptr() == stable_position_ptr)
        observed.append(shared_generation_lengths.clone())
    assert torch.equal(observed[0], observed[-1])


def test_g128_k5_spec_query_lens_v1_v2_v1_preserve_graph_storage():
    stable_query_lens = torch.empty(128, dtype=torch.int32)
    metadata = SimpleNamespace(seq_lens=stable_query_lens)
    stable_ptr = stable_query_lens.data_ptr()
    observed = []

    for verifier_budget in (256, 512, 768, 256):
        transient_query_lens = torch.full(
            (128,), verifier_budget // 128, dtype=torch.int32)
        _bind_dspark_spec_query_lens(
            metadata, transient_query_lens, 128)
        transient_query_lens.zero_()
        observed.append(metadata.seq_lens.clone())
        assert metadata.seq_lens is stable_query_lens
        assert metadata.seq_lens.data_ptr() == stable_ptr

    assert [int(lengths.sum()) for lengths in observed] == [256, 512, 768, 256]
    assert torch.equal(observed[0], observed[-1])


def test_g128_v512_physical_count_reproduces_dsa_searchsorted_oob_tail():
    retained = torch.full((128,), 3, dtype=torch.int32)
    layout = build_confidence_device_layout(
        retained,
        torch.ones(128, dtype=torch.bool),
        torch.arange(128),
        512,
        physical_draft_len=5,
    )

    wrong_req_indices = build_req_idx_per_token(layout.query_lens, 768)
    exact_req_indices = build_req_idx_per_token(layout.query_lens, 512)
    assert int(wrong_req_indices.max()) == 128
    assert int(exact_req_indices.max()) == 127
    seq_starts = torch.cumsum(layout.query_lens, 0) - layout.query_lens
    with pytest.raises(IndexError):
        _ = seq_starts[wrong_req_indices]
    assert seq_starts[exact_req_indices].shape == (512,)


@pytest.mark.parametrize("verifier_budget", [256, 512, 768])
def test_g128_exact_host_lengths_drive_actual_dsa_and_dense_topk_consumers(
    verifier_budget,
):
    retained = torch.full((128,), verifier_budget // 128 - 1,
                          dtype=torch.int32)
    exact_query_lens = retained + 1
    metadata = SimpleNamespace(
        num_tokens=verifier_budget,
        num_seqs=128,
        seq_lens=exact_query_lens,
        host_req_idx_per_token=torch.empty(verifier_budget,
                                           dtype=torch.int32),
        req_idx_per_token=torch.empty(verifier_budget, dtype=torch.int32),
        num_sparse_topk=8,
    )

    DSAtrtllmAttentionMetadata.prepare_for_indices_conversion(metadata)
    expected_requests = torch.repeat_interleave(
        torch.arange(128, dtype=torch.int32), exact_query_lens)
    assert torch.equal(metadata.req_idx_per_token, expected_requests)
    kv_lens = torch.arange(128, dtype=torch.int32) + exact_query_lens + 64
    dense_topk = DSAtrtllmAttentionMetadata._get_dense_topk_indices(
        metadata, exact_query_lens, kv_lens, verifier_budget)
    assert dense_topk.shape == (verifier_budget, 8)
    assert int(metadata.req_idx_per_token.max()) == 127


def test_g128_exact_host_lengths_bind_kv_mla_and_ratio_metadata_from_one_vector():
    exact_query_lens = torch.tensor([6] * 25 + [4] + [1] * 102,
                                    dtype=torch.int32)
    assert int(exact_query_lens.sum()) == 256
    kv_bases = torch.arange(128, dtype=torch.int32) + 100
    cached_before_prepare = kv_bases + exact_query_lens
    exact_kv_lens = cached_before_prepare + exact_query_lens
    cu_seq_lens = torch.nn.functional.pad(exact_query_lens.cumsum(0), (1, 0))

    assert int(cu_seq_lens[-1]) == 256
    assert torch.equal(exact_kv_lens,
                       kv_bases + 2 * exact_query_lens)
    for compression_ratio in (1, 2, 4, 8):
        new_compressed = (exact_kv_lens // compression_ratio -
                          cached_before_prepare // compression_ratio)
        assert new_compressed.shape == (128,)
        assert torch.all(new_compressed >= 0)
    # MLA and per-ratio decode consumers derive their generation extent from
    # the same public metadata scalar, never the physical G*(K+1) placeholder.
    assert sum(exact_query_lens.tolist()) == 256


@pytest.mark.parametrize("verifier_budget", [256, 512, 768])
def test_g128_row_map_semantics_cover_every_dynamic_tier(verifier_budget):
    retained = torch.full((128,), verifier_budget // 128 - 1,
                          dtype=torch.int32)
    slots = torch.arange(128, dtype=torch.long) + 1000
    layout = build_confidence_device_layout(
        retained,
        torch.ones(128, dtype=torch.bool),
        slots,
        verifier_budget,
        physical_draft_len=5,
        packed_capacity=768,
    )

    assert int(validate_confidence_device_layout_row_map(
        layout, 5, expected_source_slots=slots)) == 1
    assert int(layout.row_map_valid) == 1
    assert torch.count_nonzero(layout.packed_to_dense[verifier_budget:]) == 0
    assert torch.count_nonzero(
        layout.packed_draft_indices[verifier_budget - 128:]) == 0


def test_row_map_validation_rejects_interleaved_real_rows_and_stale_slots():
    interleaved_mask = torch.zeros(128, dtype=torch.bool)
    interleaved_mask[::2] = True
    retained = interleaved_mask.to(torch.int32) * 2
    interleaved = build_confidence_device_layout(
        retained,
        interleaved_mask,
        torch.arange(128),
        256,
        physical_draft_len=5,
    )
    assert int(interleaved.semantic_valid) == 1
    assert int(interleaved.row_map_valid) == 0

    real_count = 64
    slots = torch.arange(real_count, dtype=torch.long) + 200
    source_rows = torch.zeros(128, dtype=torch.long)
    source_rows[:real_count].copy_(slots)
    prefix_layout = build_confidence_device_layout(
        torch.tensor([2] * real_count + [0] * (128 - real_count),
                     dtype=torch.int32),
        torch.tensor([True] * real_count + [False] * (128 - real_count)),
        source_rows,
        256,
        physical_draft_len=5,
    )
    assert int(validate_confidence_device_layout_row_map(
        prefix_layout, 5, expected_source_slots=slots)) == 1
    assert int(validate_confidence_device_layout_row_map(
        prefix_layout, 5, expected_source_slots=slots.flip(0))) == 0


def test_row_map_validation_rejects_corrupt_packed_and_draft_maps():
    slots = torch.arange(128, dtype=torch.long)
    layout = build_confidence_device_layout(
        torch.full((128,), 3, dtype=torch.int32),
        torch.ones(128, dtype=torch.bool),
        slots,
        512,
        physical_draft_len=5,
        packed_capacity=768,
    )
    corrupt_packed = _clone_dspark_confidence_layout(layout)
    corrupt_packed.packed_to_dense[0] = 128 * 6
    assert int(validate_confidence_device_layout_row_map(
        corrupt_packed, 5, expected_source_slots=slots)) == 0

    corrupt_draft = _clone_dspark_confidence_layout(layout)
    corrupt_draft.packed_draft_indices[0] = 128 * 5
    assert int(validate_confidence_device_layout_row_map(
        corrupt_draft, 5, expected_source_slots=slots)) == 0


def test_dynamic_layout_v1_v2_v1_overwrites_all_fixed_capacity_tails():
    observed = []
    for verifier_budget in (256, 512, 256):
        layout = build_confidence_device_layout(
            torch.full((128,), verifier_budget // 128 - 1,
                       dtype=torch.int32),
            torch.ones(128, dtype=torch.bool),
            torch.arange(128),
            verifier_budget,
            physical_draft_len=5,
            packed_capacity=768,
        )
        observed.append(tuple(tensor.clone() for tensor in layout))
        assert torch.count_nonzero(
            layout.packed_to_dense[verifier_budget:]) == 0
        assert torch.count_nonzero(
            layout.packed_draft_indices[verifier_budget - 128:]) == 0

    assert all(torch.equal(first, third)
               for first, third in zip(observed[0], observed[2]))


@pytest.mark.parametrize("execution_g", [16, 32, 64, 128])
@pytest.mark.parametrize("ratio", [2, 4, 6])
def test_iteration_device_layout_matches_host_builder_for_all_profile_cells(
    execution_g, ratio
):
    verifier_budget = execution_g * ratio
    retained_lens = torch.full(
        (execution_g,), ratio - 1, dtype=torch.int32
    )
    layout = build_confidence_device_layout(
        retained_lens,
        torch.ones(execution_g, dtype=torch.bool),
        torch.arange(execution_g),
        verifier_budget,
        physical_draft_len=5,
    )
    query_lens, input_indices, draft_indices, real_tokens = (
        _build_dspark_confidence_pack_layout(
            retained_lens.tolist(), execution_g, 6, verifier_budget
        )
    )

    assert layout.query_lens.tolist() == query_lens
    assert layout.cu_query_lens.tolist() == [0] + list(
        torch.tensor(query_lens).cumsum(0).tolist()
    )
    assert layout.packed_to_dense[:real_tokens].tolist() == input_indices
    assert layout.packed_draft_indices[: sum(retained_lens)].tolist() == draft_indices
    assert torch.equal(
        layout.packed_request_ids * 6 + layout.packed_local_positions,
        layout.packed_to_dense,
    )
    assert int(layout.real_token_count) == verifier_budget
    assert int(layout.retained_token_count) == verifier_budget - execution_g
    assert int(layout.query_token_count) == verifier_budget
    assert int(layout.cu_query_token_count) == verifier_budget
    assert int(layout.real_request_count) == execution_g
    assert int(layout.semantic_valid) == 1
    assert int(layout.declared_verifier_token_budget) == verifier_budget
    assert int(layout.verifier_token_budget) == verifier_budget


def test_iteration_device_layout_dummy_suffix_and_dynamic_capacity_match_host_builder():
    retained_lens = torch.tensor([2, 1, 0, 0], dtype=torch.int32)
    layout = build_confidence_device_layout(
        retained_lens,
        torch.tensor([True, True, False, False]),
        torch.tensor([9, 12, 0, 0]),
        torch.tensor(7, dtype=torch.int32),
        physical_draft_len=5,
        packed_capacity=24,
    )
    query_lens, input_indices, draft_indices, real_tokens = (
        _build_dspark_confidence_pack_layout([2, 1, 0, 0], 2, 6, 7)
    )

    assert layout.query_lens.tolist() == query_lens
    assert layout.packed_to_dense[:real_tokens].tolist() == input_indices
    assert layout.packed_draft_indices[:3].tolist() == draft_indices
    assert layout.real_request_mask.tolist() == [True, True, False, False]
    assert layout.source_batch_indices.tolist() == [9, 12, 0, 0]
    assert int(layout.real_token_count) == real_tokens
    assert layout.packed_to_dense.shape == (24,)


def test_sampler_layout_clone_owns_every_semantic_scalar_and_tensor():
    layout = build_confidence_device_layout(
        torch.tensor([2, 1, 0, 0], dtype=torch.int32),
        torch.tensor([True, True, False, False]),
        torch.tensor([9, 12, 0, 0]),
        7,
        physical_draft_len=5,
    )
    cloned = _clone_dspark_confidence_layout(layout)

    for graph_owned, sampler_owned in zip(layout, cloned):
        assert graph_owned.data_ptr() != sampler_owned.data_ptr()
        assert torch.equal(graph_owned, sampler_owned)

    layout.retained_lens.zero_()
    layout.row_map_valid.zero_()
    layout.semantic_valid.zero_()
    layout.declared_verifier_token_budget.zero_()
    assert cloned.retained_lens.tolist() == [2, 1, 0, 0]
    assert int(cloned.row_map_valid) == 1
    assert int(cloned.semantic_valid) == 1
    assert int(cloned.declared_verifier_token_budget) == 7


def test_iteration_device_layout_v1_v2_v1_overwrites_every_visible_tail():
    destination = {
        "tokens": torch.full((24,), -1, dtype=torch.long),
        "drafts": torch.full((20,), -1, dtype=torch.long),
        "query": torch.full((4,), -1, dtype=torch.int32),
        "semantics": torch.full((7,), -1, dtype=torch.int32),
    }
    observed = []
    for retained, budget in (
        ([2, 1, 0, 0], 7),
        ([4, 3, 0, 0], 11),
        ([2, 1, 0, 0], 7),
    ):
        layout = build_confidence_device_layout(
            torch.tensor(retained, dtype=torch.int32),
            torch.tensor([True, True, False, False]),
            torch.tensor([9, 12, 0, 0]),
            budget,
            physical_draft_len=5,
        )
        for tensor in destination.values():
            tensor.zero_()
        destination["tokens"][:budget].copy_(layout.packed_to_dense)
        destination["drafts"][: budget - 4].copy_(layout.packed_draft_indices)
        destination["query"].copy_(layout.query_lens)
        destination["semantics"].copy_(torch.stack((
            layout.semantic_valid,
            layout.row_map_valid,
            layout.retained_token_count,
            layout.query_token_count,
            layout.cu_query_token_count,
            layout.real_request_count,
            layout.declared_verifier_token_budget,
        )))
        observed.append(tuple(tensor.clone() for tensor in destination.values()))

    assert all(torch.equal(first, third) for first, third in zip(observed[0], observed[2]))
    assert torch.count_nonzero(observed[2][0][7:]) == 0
    assert torch.count_nonzero(observed[2][1][3:]) == 0
    assert observed[0][3].tolist() == [1, 1, 3, 7, 7, 2, 7]
    assert observed[1][3].tolist() == [1, 1, 7, 11, 11, 2, 11]


def test_event_owned_host_query_lens_v1_v2_v1_overwrites_without_aliasing():
    observed = []
    buffers = []
    for verifier_budget in (256, 512, 256):
        layout = build_confidence_device_layout(
            torch.full((128,), verifier_budget // 128 - 1,
                       dtype=torch.int32),
            torch.ones(128, dtype=torch.bool),
            torch.arange(128),
            verifier_budget,
            physical_draft_len=5,
            packed_capacity=768,
        )
        host_query_lens = torch.empty(128, dtype=torch.int32)
        host_query_lens.copy_(layout.query_lens)
        buffers.append(host_query_lens)
        observed.append(host_query_lens.clone())
        layout.query_lens.zero_()
        assert int(host_query_lens.sum()) == verifier_budget

    assert buffers[0].data_ptr() != buffers[1].data_ptr()
    assert buffers[1].data_ptr() != buffers[2].data_ptr()
    assert torch.equal(observed[0], observed[2])
    assert not torch.equal(observed[0], observed[1])


def _iteration_route_runner(config, *, last_epoch=0):
    dist = Mock()
    dist.tp_allgather.side_effect = lambda payload: [payload] * 2
    return SimpleNamespace(
        enabled=True,
        padding_enabled=True,
        max_supported_batch_size=32,
        supported_batch_sizes=[16, 32],
        spec_config=config,
        confidence_engine_generation=17,
        _confidence_last_route_epoch=last_epoch,
        _capture_allowed=False,
        config=SimpleNamespace(
            enable_attention_dp=True,
            dist=dist,
            batch_size=32,
        ),
    )


def test_production_order_first_step_full_k_then_carried_g16_v32_compact_key():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, None
    ) == 16
    assert runner.confidence_force_full_k_route
    assert runner.confidence_adp_verifier_token_budget == 0

    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier
    ) == 16
    assert runner.confidence_device_layout is carrier.dspark_confidence_layout
    assert runner.confidence_adp_verifier_token_budget == 32
    assert runner.confidence_adp_route_epoch == 1

    key_runner = runner_stub(config)
    key_runner.confidence_device_layout = carrier.dspark_confidence_layout
    key = CUDAGraphRunner.get_graph_key(
        key_runner,
        batch,
        new_tensors_device=carrier,
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
            confidence_force_full_k_route=False,
            confidence_verifier_token_budget=32,
        ),
    )
    assert (key.batch_size, key.draft_len, key.verifier_num_tokens) == (16, 5, 32)
    assert carrier.dspark_confidence_layout.query_lens.sum() == 32


def test_carried_layout_roster_binds_seq_slots_not_transient_batch_indices():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    requests = [
        request_stub(index, 5, index, seq_slot=32 + index)
        for index in range(16)
    ]
    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier) == 16
    assert runner.confidence_device_layout is carrier.dspark_confidence_layout
    assert carrier.dspark_confidence_layout.source_batch_indices.tolist() == list(
        range(32, 48))


def test_g128_v256_undercounted_device_mask_discards_to_v0_before_packing():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={128: 256},
    )
    requests = [request_stub(index, 5, index) for index in range(128)]
    retained = torch.tensor([5] * 17 + [0] * 111, dtype=torch.int32)
    real_mask = torch.zeros(128, dtype=torch.bool)
    real_mask[:17] = True
    layout = build_confidence_device_layout(
        retained,
        real_mask,
        torch.arange(128),
        256,
        physical_draft_len=5,
    )
    carrier = _layout_carrier(
        requests,
        retained.tolist(),
        execution_g=128,
        verifier_budget=256,
        route_epoch=1,
    )
    carrier.dspark_confidence_layout = layout
    carrier.dspark_confidence_verifier_token_budget_host = (
        layout.verifier_token_budget.clone())
    carrier.dspark_confidence_semantics_host = torch.stack((
        layout.semantic_valid,
        layout.row_map_valid,
        layout.retained_token_count,
        layout.query_token_count,
        layout.cu_query_token_count,
        layout.real_request_count,
        layout.declared_verifier_token_budget,
    )).clone()
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)
    runner.max_supported_batch_size = 128
    runner.supported_batch_sizes = [128]
    runner.config.batch_size = 128

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier) == 0
    assert runner.confidence_force_full_k_route
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_device_layout is None
    assert runner.confidence_adp_verifier_token_budget == 0
    assert all(request.py_draft_tokens_effective_len == 5
               for request in requests)


def test_empty_adp_peer_discards_infeasible_compact_carrier_to_shared_full_k():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    carrier = _layout_carrier(
        [],
        [0] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    batch = ScheduledRequests()
    batch.generation_requests = []
    runner = _iteration_route_runner(config)

    peer_route = _adp_semantic_route(
        16,
        0,
        5,
        1,
        3,
        layout_shapes_ready=False,
        base_layout_ready=True,
        semantic_exact=False,
        semantic_valid=0,
        row_map_valid=0,
        retained_count=15,
        query_count=28,
        cu_query_count=28,
        declared_v=32,
    )

    def asymmetric_tail(payload):
        if len(payload) == len(_ADP_SEMANTIC_ROUTE_FIELDS):
            return [payload, peer_route]
        if len(payload) == 2:
            return [payload, [True, 3]]
        assert len(payload) == 3
        return [payload, [True, 3, False]]

    runner.config.dist.tp_allgather.side_effect = asymmetric_tail

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier
    ) == 0
    assert runner.confidence_force_full_k_route
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_device_layout is None


def test_semantic_tuple_asymmetry_discards_to_shared_full_k():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    def asymmetric_semantics(payload):
        peer = list(payload)
        peer[2] = False
        peer[10] -= 1
        return [payload, peer]

    runner.config.dist.tp_allgather.side_effect = asymmetric_semantics

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier) == 0
    assert runner.confidence_force_full_k_route
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_device_layout is None


def test_stale_row_map_semantics_discards_before_compact_packing():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    carrier.dspark_confidence_semantics_host[1] = 0
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier) == 0
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_force_full_k_route
    assert runner.confidence_device_layout is None


def test_row_map_semantic_asymmetry_discards_to_shared_full_k():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    def asymmetric_row_map(payload):
        peer = list(payload)
        peer[2] = False
        peer[9] = 0
        return [payload, peer]

    runner.config.dist.tp_allgather.side_effect = asymmetric_row_map
    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier) == 0
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_force_full_k_route
    assert runner.confidence_device_layout is None


def test_carried_physical_full_k_layout_ignores_stale_host_compact_lengths():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={128: 768},
    )
    requests = [request_stub(index, 1, index) for index in range(128)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    carrier = _layout_carrier(
        requests,
        [5] * 128,
        execution_g=128,
        verifier_budget=768,
        route_epoch=1,
    )
    runner = runner_stub(config)
    runner.confidence_device_layout = carrier.dspark_confidence_layout

    key = CUDAGraphRunner.get_graph_key(
        runner,
        batch,
        new_tensors_device=carrier,
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=True,
            confidence_force_full_k_route=False,
            confidence_verifier_token_budget=768,
        ),
    )

    assert key.verifier_num_tokens == 768
    assert carrier.dspark_confidence_layout.query_lens.tolist() == [6] * 128


def test_full_k_carrier_roster_asymmetry_clears_stale_compact_request_state():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 96},
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    for request in requests:
        request.py_draft_tokens_effective_len = 1
        request.py_dspark_confidence_route_epoch = 1
        request.py_dspark_confidence_execution_batch_size = 16
        request.py_dspark_confidence_verifier_token_budget = 32
    carrier = _layout_carrier(
        requests,
        [5] * 16,
        execution_g=16,
        verifier_budget=96,
        route_epoch=2,
    )
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    def peer_completed_one_request(payload):
        if len(payload) == 2:
            return [payload, [True, 15]]
        peer = list(payload)
        peer[0] = False
        peer[1] = False
        peer[7] = 15
        peer[13] = 15
        return [payload, peer]

    runner.config.dist.tp_allgather.side_effect = peer_completed_one_request
    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier
    ) == 16
    assert runner.confidence_force_full_k_route
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_device_layout is None
    assert runner.confidence_adp_execution_batch_size == 16
    assert all(request.py_draft_tokens_effective_len == 5 for request in requests)
    assert all(
        request.py_dspark_confidence_route_epoch is None
        and request.py_dspark_confidence_execution_batch_size is None
        and request.py_dspark_confidence_verifier_token_budget is None
        for request in requests
    )


def test_context_batch_has_no_compact_route_or_plan():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )
    runner = _iteration_route_runner(config)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request_stub(900, 5, None)]
    batch.generation_requests = [
        request_stub(index, 5, index) for index in range(17)
    ]

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, None
    ) == 0
    assert runner.confidence_force_full_k_route
    assert runner.confidence_adp_verifier_token_budget == 0


def test_warmup17_promoted_generation_none_effective_len_uses_g32_v0_full_k():
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32, 32: 64},
    )
    runner = _iteration_route_runner(config)
    requests = [request_stub(index, None, index) for index in range(17)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    unplanned_inputs = SampleStateTensorsSpec(
        new_tokens=torch.empty((6, 17, 1), dtype=torch.int32),
        new_tokens_lens=None,
        next_draft_tokens=torch.zeros((17, 5), dtype=torch.int32),
    )

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, unplanned_inputs
    ) == 32
    assert runner.confidence_force_full_k_route
    assert runner.confidence_adp_execution_batch_size == 32
    assert runner.confidence_adp_verifier_token_budget == 0

    padding_requests = [
        request_stub(900 + index, 5, None, dummy=True) for index in range(15)
    ]
    for request in padding_requests:
        request.is_cuda_graph_dummy = True
    batch.generation_requests.extend(padding_requests)

    key_runner = runner_stub(config)
    full_key = CUDAGraphRunner.get_graph_key(
        key_runner,
        batch,
        new_tensors_device=unplanned_inputs,
        spec_metadata=SimpleNamespace(
            is_all_greedy_sample=True,
            confidence_fixed_budget_active=False,
            confidence_force_full_k_route=True,
            confidence_verifier_token_budget=0,
        ),
    )
    assert (full_key.batch_size, full_key.draft_len, full_key.verifier_num_tokens) == (
        32,
        5,
        0,
    )

    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
    engine.cuda_graph_runner = runner
    engine.spec_config = config
    assert engine._align_dspark_confidence_lengths_to_graph(
        batch, full_key, new_tensors_device=unplanned_inputs
    )
    assert all(
        request.py_draft_tokens_effective_len == 5
        for request in batch.generation_requests
    )

    requests[0].py_draft_tokens_effective_len = 4
    with pytest.raises(RuntimeError, match="cannot be reconstructed as full-K"):
        engine._align_dspark_confidence_lengths_to_graph(
            batch, full_key, new_tensors_device=unplanned_inputs
        )


@pytest.mark.parametrize("failure", ["stale", "engine", "asymmetric"])
def test_compact_carrier_stale_or_asymmetric_epoch_fails_closed(failure):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config, last_epoch=4 if failure == "stale" else 0)
    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=4,
    )
    if failure == "engine":
        carrier.dspark_confidence_engine_generation = 99
    if failure == "asymmetric":
        calls = 0

        def asymmetric(payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                peer = list(payload)
                assert len(peer) == len(_ADP_SEMANTIC_ROUTE_FIELDS) == 15
                _set_adp_semantic_route_field(peer, "carried_v", 31)
                _set_adp_semantic_route_field(peer, "declared_v", 31)
                return [payload, peer]
            return [payload] * 2

        runner.config.dist.tp_allgather.side_effect = asymmetric

    with pytest.raises(RuntimeError, match="provenance is missing, stale, or asymmetric"):
        CUDAGraphRunner._get_confidence_adp_common_batch_size(
            runner, batch, carrier
        )
    assert runner.confidence_force_full_k_route


@pytest.mark.parametrize("roster_change", ["completion", "reorder"])
def test_completed_or_reordered_carrier_roster_discards_to_shared_full_k(
    roster_change,
):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    planned = [request_stub(index, 5, index) for index in range(16)]
    carrier = _layout_carrier(
        planned,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    current = planned[:-1] if roster_change == "completion" else list(reversed(planned))
    for request in current:
        request.py_draft_tokens_effective_len = 1
        request.py_dspark_confidence_route_epoch = 1
        request.py_dspark_confidence_execution_batch_size = 16
        request.py_dspark_confidence_verifier_token_budget = 32
    batch = ScheduledRequests()
    batch.generation_requests = current
    runner = _iteration_route_runner(config)

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier
    ) == 16
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_force_full_k_route
    assert runner.confidence_device_layout is None
    assert all(request.py_draft_tokens_effective_len == 5 for request in current)


def test_accept_draft_tokens_preserves_exact_iteration_layout_identity():
    requests = [request_stub(index, 5, index) for index in range(2)]
    carrier = _layout_carrier(
        requests,
        [4, 3],
        execution_g=2,
        verifier_budget=9,
        route_epoch=3,
    )
    budget_event = object()
    carrier.dspark_confidence_budget_ready_event = budget_event
    target_inputs = SampleStateTensorsSpec(
        new_tokens=torch.empty((6, 2, 1), dtype=torch.int32),
        new_tokens_lens=torch.ones(2, dtype=torch.int32),
        next_draft_tokens=torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.int32
        ),
        **carrier.__dict__,
    )
    target_outputs = SimpleNamespace(
        new_tokens=torch.tensor(
            [
                [[1], [6]],
                [[2], [99]],
                [[3], [98]],
                [[4], [97]],
                [[5], [96]],
                [[11], [12]],
            ],
            dtype=torch.int32,
        ),
        log_probs=None,
    )
    batch = ScheduledRequests()
    batch.generation_requests = requests

    accepted, _ = PyExecutor._accept_draft_tokens(
        object.__new__(PyExecutor), batch, target_outputs, target_inputs
    )

    assert accepted.dspark_confidence_layout is target_inputs.dspark_confidence_layout
    assert accepted.next_draft_lens is target_inputs.next_draft_lens
    assert (accepted.dspark_confidence_semantics_host is
            target_inputs.dspark_confidence_semantics_host)
    assert accepted.dspark_confidence_budget_ready_event is budget_event
    assert (accepted.dspark_confidence_query_lens_host is
            target_inputs.dspark_confidence_query_lens_host)
    assert accepted.dspark_confidence_route_epoch == 3
    assert accepted.dspark_confidence_request_ids == (0, 1)
    assert accepted.dspark_confidence_seq_slots == (0, 1)
    assert accepted.dspark_confidence_engine_generation == 17


def test_device_v_and_semantics_wait_only_on_dedicated_event_at_keying():
    class BudgetEvent:

        def __init__(self):
            self.waits = 0

        def synchronize(self):
            self.waits += 1

    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="dynamic_budget",
        confidence_verifier_token_budget_tiers={16: [32, 64]},
        confidence_sps_cost_table_path="/tmp/dummy-sps.json",
    )
    requests = [request_stub(index, 5, index) for index in range(16)]
    carrier = _layout_carrier(
        requests,
        [1] * 16,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    budget_event = BudgetEvent()
    carrier.dspark_confidence_verifier_token_budget = None
    carrier.dspark_confidence_verifier_token_budget_host = torch.tensor(
        32, dtype=torch.int32
    )
    carrier.dspark_confidence_budget_ready_event = budget_event
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    assert budget_event.waits == 0
    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier
    ) == 16
    assert budget_event.waits == 1
    assert runner.confidence_adp_verifier_token_budget == 32
    assert (runner.confidence_query_lens_host is
            carrier.dspark_confidence_query_lens_host)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "wrong_g", "real_zero", "real_over_k", "dummy_not_one", "wrong_sum"],
)
def test_invalid_event_owned_host_query_lens_discards_same_provenance_to_v0(
    mutation,
):
    config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model="/tmp/dummy_model",
        confidence_mode="fixed_budget",
        confidence_verifier_token_budget_schedule={16: 32},
    )
    requests = [request_stub(index, 5, index) for index in range(8)]
    carrier = _layout_carrier(
        requests,
        [2] * 8 + [0] * 8,
        execution_g=16,
        verifier_budget=32,
        route_epoch=1,
    )
    query_lens = carrier.dspark_confidence_query_lens_host
    if mutation == "missing":
        carrier.dspark_confidence_query_lens_host = None
    elif mutation == "wrong_g":
        carrier.dspark_confidence_query_lens_host = query_lens[:-1]
    elif mutation == "real_zero":
        query_lens[0] = 0
    elif mutation == "real_over_k":
        query_lens[0] = 7
    elif mutation == "dummy_not_one":
        query_lens[-1] = 2
    else:
        query_lens[0] += 1
    batch = ScheduledRequests()
    batch.generation_requests = requests
    runner = _iteration_route_runner(config)

    assert CUDAGraphRunner._get_confidence_adp_common_batch_size(
        runner, batch, carrier) == 0
    assert runner.confidence_force_full_k_route
    assert runner.confidence_discarded_device_layout
    assert runner.confidence_device_layout is None
    assert runner.confidence_query_lens_host is None
    assert all(request.py_draft_tokens_effective_len == 5
               for request in requests)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA replay storage")
@pytest.mark.parametrize(("execution_g", "ratio"),
                         [(16, 2), (32, 4), (64, 6), (128, 2),
                          (128, 4), (128, 6)])
def test_cuda_device_layout_replay_matches_host_packing_and_model_extent(
    execution_g, ratio
):
    verifier_budget = execution_g * ratio
    retained = torch.full(
        (execution_g,), ratio - 1, dtype=torch.int32, device="cuda"
    )
    layout = build_confidence_device_layout(
        retained,
        torch.ones(execution_g, dtype=torch.bool, device="cuda"),
        torch.arange(execution_g, device="cuda"),
        verifier_budget,
        physical_draft_len=5,
    )
    dense = torch.arange(execution_g * 6, device="cuda")
    packed = dense.index_select(0, layout.packed_to_dense)
    _, host_indices, _, _ = _build_dspark_confidence_pack_layout(
        [ratio - 1] * execution_g, execution_g, 6, verifier_budget
    )

    torch.cuda.synchronize()
    assert packed.cpu().tolist() == dense.cpu()[host_indices].tolist()
    assert int(layout.query_lens.sum().cpu()) == verifier_budget
    assert layout.cu_query_lens[-1].cpu() == verifier_budget
    assert int(layout.row_map_valid.cpu()) == 1
    assert int(layout.semantic_valid.cpu()) == 1
    assert int(layout.retained_token_count.cpu()) == verifier_budget - execution_g
    assert int(layout.query_token_count.cpu()) == verifier_budget
    assert int(layout.cu_query_token_count.cpu()) == verifier_budget
    assert int(layout.real_request_count.cpu()) == execution_g
    assert int(layout.declared_verifier_token_budget.cpu()) == verifier_budget
    assert int(layout.verifier_token_budget.cpu()) == verifier_budget


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA async D2H/event storage")
def test_cuda_sampler_owned_query_lens_event_v1_v2_v1_has_no_graph_alias():
    observed = []
    host_buffers = []
    for verifier_budget in (256, 512, 256):
        graph_layout = build_confidence_device_layout(
            torch.full((128,),
                       verifier_budget // 128 - 1,
                       dtype=torch.int32,
                       device="cuda"),
            torch.ones(128, dtype=torch.bool, device="cuda"),
            torch.arange(128, device="cuda"),
            verifier_budget,
            physical_draft_len=5,
            packed_capacity=768,
        )
        sampler_layout = _clone_dspark_confidence_layout(graph_layout)
        host_query_lens = torch.empty(128,
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=True)
        host_query_lens.copy_(sampler_layout.query_lens, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()
        graph_layout.query_lens.zero_()
        torch.cuda.synchronize()
        host_buffers.append(host_query_lens)
        observed.append(host_query_lens.clone())
        assert host_query_lens.is_pinned()
        assert int(host_query_lens.sum()) == verifier_budget

    assert host_buffers[0].data_ptr() != host_buffers[1].data_ptr()
    assert host_buffers[1].data_ptr() != host_buffers[2].data_ptr()
    assert torch.equal(observed[0], observed[2])
    assert not torch.equal(observed[0], observed[1])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA attention metadata storage")
def test_cuda_shared_cross_tier_generation_buffer_v256_v512_v768_v256():
    base_metadata = _cpu_attention_metadata_with_physical_g128_k5()
    base_metadata.spec_decoding_generation_lengths = torch.empty(
        128, dtype=torch.int32, device="cuda")
    base_metadata.spec_decoding_position_offsets = torch.arange(
        6, dtype=torch.int32, device="cuda").repeat(128)
    graph_metadata = {
        verifier_budget: copy.copy(base_metadata)
        for verifier_budget in (256, 512, 768)
    }
    shared_generation_lengths = base_metadata.spec_decoding_generation_lengths
    stable_generation_ptr = shared_generation_lengths.data_ptr()
    stable_position_ptr = base_metadata.spec_decoding_position_offsets.data_ptr()
    observed = []

    for verifier_budget in (256, 512, 768, 256):
        live_query_lens = (
            _g128_k5_ragged_retained_lens(
                verifier_budget, device="cuda") + 1)
        _refresh_dspark_confidence_graph_generation_lengths(
            graph_metadata[verifier_budget], live_query_lens, 128)
        torch.cuda.synchronize()

        assert shared_generation_lengths.data_ptr() == stable_generation_ptr
        assert (graph_metadata[verifier_budget].
                spec_decoding_position_offsets.data_ptr()
                == stable_position_ptr)
        assert torch.equal(shared_generation_lengths, live_query_lens)
        assert int(shared_generation_lengths.sum().cpu()) == verifier_budget
        observed.append(shared_generation_lengths.cpu())
    assert torch.equal(observed[0], observed[-1])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA attention metadata storage")
@pytest.mark.parametrize("verifier_budget", [256, 512, 768])
def test_cuda_g128_exact_attention_bind_survives_final_host_setter(
    verifier_budget,
):
    metadata = AttentionMetadata(
        max_num_requests=128,
        max_num_tokens=768,
        is_cuda_graph=True,
    )
    metadata.seq_lens = torch.full((128,), 6, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.confidence_fixed_budget_active = True
    metadata.spec_decoding_generation_lengths = torch.empty(
        128, dtype=torch.int32, device="cuda")
    metadata.spec_decoding_position_offsets = torch.arange(
        6, dtype=torch.int32, device="cuda").repeat(128)
    stable_host = metadata.seq_lens
    stable_host_ptr = stable_host.data_ptr()
    stable_device_ptr = metadata.seq_lens_cuda.data_ptr()
    stable_generation_ptr = metadata.spec_decoding_generation_lengths.data_ptr()
    stable_position_ptr = metadata.spec_decoding_position_offsets.data_ptr()
    layout = build_confidence_device_layout(
        _g128_k5_ragged_retained_lens(verifier_budget, device="cuda"),
        torch.ones(128, dtype=torch.bool, device="cuda"),
        torch.arange(128, device="cuda"),
        verifier_budget,
        physical_draft_len=5,
        packed_capacity=768,
    )

    assert metadata.num_tokens == 768
    exact_host_query_lens = layout.query_lens.cpu()
    _bind_dspark_confidence_attention_layout(
        metadata, layout, exact_host_query_lens, 128, verifier_budget)
    torch.cuda.synchronize()
    assert metadata.num_tokens == verifier_budget
    assert metadata.seq_lens.tolist() == exact_host_query_lens.tolist()
    assert int(metadata.seq_lens_cuda.sum().cpu()) == verifier_budget
    assert metadata.seq_lens is stable_host
    assert metadata.seq_lens.data_ptr() == stable_host_ptr
    assert metadata.seq_lens_cuda.data_ptr() == stable_device_ptr
    assert (metadata.spec_decoding_generation_lengths.data_ptr()
            == stable_generation_ptr)
    assert metadata.spec_decoding_position_offsets.data_ptr() == stable_position_ptr
    assert torch.equal(metadata.spec_decoding_generation_lengths,
                       layout.query_lens)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA sequence-length storage")
def test_cuda_pageable_graph_host_lengths_update_in_place_v1_v2_v1():
    metadata = _cpu_attention_metadata_with_physical_g128_k5()
    metadata._seq_lens_cuda = torch.empty(
        128, dtype=torch.int32, device="cuda")
    metadata.spec_decoding_generation_lengths = torch.empty(
        128, dtype=torch.int32, device="cuda")
    metadata.spec_decoding_position_offsets = torch.arange(
        6, dtype=torch.int32, device="cuda").repeat(128)
    stable_host = metadata.seq_lens
    stable_host_ptr = stable_host.data_ptr()
    stable_device_ptr = metadata.seq_lens_cuda.data_ptr()
    assert not stable_host.is_pinned()
    observed = []

    for verifier_budget in (256, 512, 768, 256):
        retained = _g128_k5_ragged_retained_lens(
            verifier_budget, device="cuda")
        layout = build_confidence_device_layout(
            retained,
            torch.ones(128, dtype=torch.bool, device="cuda"),
            torch.arange(128, device="cuda"),
            verifier_budget,
            physical_draft_len=5,
            packed_capacity=768,
        )
        event_owned_host = layout.query_lens.cpu()
        _bind_dspark_confidence_attention_layout(
            metadata, layout, event_owned_host, 128, verifier_budget)
        event_owned_host.zero_()
        torch.cuda.synchronize()

        assert metadata.seq_lens is stable_host
        assert metadata.seq_lens.data_ptr() == stable_host_ptr
        assert metadata.seq_lens_cuda.data_ptr() == stable_device_ptr
        assert metadata.num_tokens == verifier_budget
        assert int(metadata.seq_lens_cuda.sum().cpu()) == verifier_budget
        observed.append(metadata.seq_lens.clone())

    assert torch.equal(observed[0], observed[-1])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA graph-owned DSpark metadata")
def test_cuda_dspark_query_and_draft_lens_keep_graph_pointers_v1_v2_v1():
    base_metadata = DSparkSpecMetadata(
        max_num_requests=128,
        max_draft_len=5,
        max_total_draft_tokens=5,
    )
    metadata = base_metadata.create_cuda_graph_metadata(128)
    assert metadata.is_cuda_graph
    stable_query_ptr = metadata.seq_lens.data_ptr()
    stable_draft_ptr = metadata.draft_lens.data_ptr()
    observed = []

    for verifier_budget in (256, 512, 768, 256):
        retained = torch.full((128,),
                              verifier_budget // 128 - 1,
                              dtype=torch.int32,
                              device="cuda")
        transient_query_lens = retained + 1
        _bind_dspark_spec_query_lens(metadata, transient_query_lens, 128)
        metadata.draft_lens.copy_(retained)
        transient_query_lens.zero_()
        torch.cuda.synchronize()

        assert metadata.seq_lens.data_ptr() == stable_query_ptr
        assert metadata.draft_lens.data_ptr() == stable_draft_ptr
        assert int(metadata.seq_lens.sum().cpu()) == verifier_budget
        assert int(metadata.draft_lens.sum().cpu()) == verifier_budget - 128
        observed.append(metadata.seq_lens.cpu())

    assert torch.equal(observed[0], observed[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA acceptance storage")
def test_cuda_compact_acceptance_and_kv_rewind_match_host_reference():
    verified = torch.tensor([4, 3, 1, 0], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([2, 3, 0, 0], dtype=torch.int32, device="cuda")
    query_lens = verified + 1
    new_tokens_lens = accepted + 1
    rewind = verified - accepted
    kv_offsets = new_tokens_lens - query_lens

    torch.cuda.synchronize()
    assert rewind.cpu().tolist() == [2, 0, 1, 0]
    assert kv_offsets.cpu().tolist() == [-2, 0, -1, 0]
    assert torch.all(rewind >= 0)
