# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph identity and fallback tests for DSpark confidence tiers."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig


def request_stub(request_id, effective_len, batch_idx, *, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id,
        py_draft_tokens=[11, 12, 13, 14, 15],
        py_draft_tokens_effective_len=effective_len,
        py_batch_idx=batch_idx,
        is_dummy=dummy,
    )


def runner_stub(config, *, capture=False):
    runner = Mock()
    runner.config = SimpleNamespace(is_draft_model=False)
    runner.max_beam_width = 1
    runner._capture_allowed = capture
    runner.spec_config = config
    runner._get_seq_len_mode.return_value = False
    return runner


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


def test_attention_dp_v_mismatch_uses_full_k_on_every_rank():
    runner = Mock()
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
        [True, 2, 9, 5],
        [True, 2, 12, 5],
    ]
    batch = ScheduledRequests()
    batch.generation_requests = [
        request_stub(1, 4, 0),
        request_stub(2, 3, 1),
    ]
    with patch(
        "tensorrt_llm._torch.pyexecutor.cuda_graph_runner.ExpertStatistic.should_record",
        return_value=False,
    ):
        result = CUDAGraphRunner.maybe_get_cuda_graph(
            runner,
            batch,
            enable_spec_decode=True,
            attn_metadata=object(),
            new_tensors_device=SimpleNamespace(),
        )
    assert result == (graph_attn_metadata, graph_spec_metadata, full_key)


def test_full_k_fallback_realigns_request_state():
    engine = object.__new__(PyTorchModelEngine)
    engine.is_draft_model = False
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
    full_key = KeyType(batch_size=2, draft_len=5, is_first_draft=False)
    assert engine._align_dspark_confidence_lengths_to_graph(batch, full_key)
    assert [r.py_draft_tokens_effective_len for r in batch.generation_requests] == [5, 5]


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
