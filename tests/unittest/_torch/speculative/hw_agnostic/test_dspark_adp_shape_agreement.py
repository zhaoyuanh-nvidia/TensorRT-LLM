# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the cached DSpark attention-DP shape agreement."""

import dataclasses
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import ADPShapeAgreement, CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import (
    _DSPARK_ADP_WIRE_TRAILER_LEN,
    _DSPARK_EXACT_NATIVE_YIELD_INDEX,
    _DSPARK_EXACT_WIRE_PREFIX_LEN,
    PyExecutor,
    _decode_dspark_exact_expected_yields,
    _validate_dspark_adp_acceptance_gate,
    _validate_dspark_adp_debug_flags,
    _validate_dspark_exact_bucket,
)


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


@pytest.mark.parametrize(
    ("trims", "bucket"),
    [(True, 22), (False, None)],
)
def test_exact_bucket_invariant_accepts_compact_and_cap_accept(trims, bucket):
    _validate_dspark_exact_bucket(
        exact_shape=(4, 22, 4),
        bucket=bucket,
        trims_submitted_tokens=trims,
    )


@pytest.mark.parametrize(
    ("trims", "bucket", "message"),
    [
        (True, None, "compact policy"),
        (True, 21, "compact policy"),
        (False, 22, "cap-accept policy"),
    ],
)
def test_exact_bucket_invariant_rejects_wrong_graph_key(trims, bucket, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_dspark_exact_bucket(
            exact_shape=(4, 22, 4),
            bucket=bucket,
            trims_submitted_tokens=trims,
        )


@pytest.mark.parametrize(("trims", "bucket"), [(True, 22), (False, None)])
def test_non_exact_policy_has_no_bucket_constraint(trims, bucket):
    _validate_dspark_exact_bucket(
        exact_shape=None,
        bucket=bucket,
        trims_submitted_tokens=trims,
    )


@pytest.mark.parametrize(
    ("confidence", "attention_dp", "gate"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_acceptance_gate_validation_is_noop_when_a_feature_is_off(confidence, attention_dp, gate):
    _validate_dspark_adp_acceptance_gate(
        confidence_enabled=confidence,
        attention_dp_enabled=attention_dp,
        speculation_gate_enabled=gate,
    )


def test_acceptance_gate_validation_rejects_rank_local_adp_disable():
    with pytest.raises(ValueError, match="acceptance_rate_window_size"):
        _validate_dspark_adp_acceptance_gate(
            confidence_enabled=True,
            attention_dp_enabled=True,
            speculation_gate_enabled=True,
        )


def test_debug_flag_uses_the_final_adp_payload_field():
    exact_cells = 2
    payload_len = _DSPARK_EXACT_WIRE_PREFIX_LEN + exact_cells + _DSPARK_ADP_WIRE_TRAILER_LEN
    debug_index = payload_len - 1
    assert payload_len == 22 + exact_cells
    assert debug_index == 21 + exact_cells
    payloads = [[0] * payload_len for _ in range(2)]
    payloads[0][debug_index] = payloads[1][debug_index] = 1
    assert _validate_dspark_adp_debug_flags(payloads, debug_index)


def test_exact_yield_decode_stops_before_three_field_adp_trailer():
    exact_cells = ((16, 64), (128, 704))
    payload_len = _DSPARK_EXACT_WIRE_PREFIX_LEN + len(exact_cells) + _DSPARK_ADP_WIRE_TRAILER_LEN
    payloads = [[0] * payload_len for _ in range(2)]
    payloads[0][_DSPARK_EXACT_NATIVE_YIELD_INDEX] = 10_000_000
    payloads[1][_DSPARK_EXACT_NATIVE_YIELD_INDEX] = 20_000_000
    payloads[0][_DSPARK_EXACT_WIRE_PREFIX_LEN] = 3_000_000
    payloads[1][_DSPARK_EXACT_WIRE_PREFIX_LEN] = 4_000_000
    payloads[0][_DSPARK_EXACT_WIRE_PREFIX_LEN + 1] = 5_000_000
    payloads[1][_DSPARK_EXACT_WIRE_PREFIX_LEN + 1] = 6_000_000
    # C's trailer starts immediately after the exact cells. Its batch-size,
    # padding-ready, and debug values must not be interpreted as yields.
    trailer_start = _DSPARK_EXACT_WIRE_PREFIX_LEN + len(exact_cells)
    payloads[0][trailer_start:] = [128, 1, 1]
    payloads[1][trailer_start:] = [64, 0, 1]

    native_yield, compact_yields = _decode_dspark_exact_expected_yields(
        payloads=payloads,
        exact_cells=exact_cells,
        graph_batch_size=128,
        yield_scale=1_000_000,
    )

    assert payload_len == 22 + len(exact_cells)
    assert native_yield == 30.0
    assert compact_yields == {704: 11.0}


def test_exact_yield_decode_fails_closed_on_any_rank_sentinel():
    exact_cells = ((128, 704),)
    payload_len = _DSPARK_EXACT_WIRE_PREFIX_LEN + len(exact_cells) + _DSPARK_ADP_WIRE_TRAILER_LEN
    payloads = [[0] * payload_len for _ in range(2)]
    payloads[0][_DSPARK_EXACT_WIRE_PREFIX_LEN] = 5_000_000
    payloads[1][_DSPARK_EXACT_WIRE_PREFIX_LEN] = -1

    _, compact_yields = _decode_dspark_exact_expected_yields(
        payloads=payloads,
        exact_cells=exact_cells,
        graph_batch_size=128,
        yield_scale=1_000_000,
    )

    assert compact_yields == {704: 0.0}


def test_debug_flag_disagreement_fails_identically_after_policy_collective():
    debug_index = _DSPARK_EXACT_WIRE_PREFIX_LEN + 2
    payloads = [[0] * (debug_index + 1) for _ in range(2)]
    payloads[1][debug_index] = 1
    with pytest.raises(RuntimeError, match="collective ordering"):
        _validate_dspark_adp_debug_flags(payloads, debug_index)


def test_debug_flag_rejects_non_boolean_wire_value():
    with pytest.raises(RuntimeError, match="encoded as 0 or 1"):
        _validate_dspark_adp_debug_flags([[2]], 0)


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


def test_confidence_off_graph_lookup_preserves_two_field_adp_protocol():
    """Static K5 must not pay the DSpark shape-agreement payload or scan."""
    payloads = []

    def gather(value):
        payloads.append(value)
        return [[True, 3], [True, 4]]

    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    runner.enabled = True
    runner._dspark_confidence_enabled = False
    runner.adp_shape_agreement = object()
    runner.adp_shape_debug = False
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        mapping=SimpleNamespace(tp_size=8),
        dist=SimpleNamespace(tp_allgather=gather),
    )
    runner._is_mixed_encoder_decoder_batch = lambda _batch: False
    runner._can_run_cuda_graph_batch = lambda _batch: True
    runner._local_draft_len = lambda _batch: (_ for _ in ()).throw(
        AssertionError("confidence-off graph lookup scanned draft lengths"))

    assert runner.maybe_get_cuda_graph(_Batch([object()] * 3), False,
                                       object()) == (None, None, None)
    assert payloads == [[True, 3]]
    assert runner.last_miss_reason == "peer_shape_mismatch"


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


def test_releasing_padding_dummy_invalidates_cached_agreement():
    batch = _Batch([object()] * 3)
    dummy = object()
    freed = []
    manager = SimpleNamespace(free_resources=freed.append)
    runner = SimpleNamespace(
        padding_dummy_requests={5: dummy},
        adp_shape_agreement=_agreement(batch),
        _padding_dummy_managers=lambda _resource_manager: [manager],
    )

    assert CUDAGraphRunner.release_padding_dummy(runner, object(), 5)
    assert runner.adp_shape_agreement is None
    assert freed == [dummy]
