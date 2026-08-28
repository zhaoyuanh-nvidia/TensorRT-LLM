# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the cached DSpark attention-DP shape agreement."""

import dataclasses
from types import SimpleNamespace

import pytest
import tensorrt_llm._torch.pyexecutor.cuda_graph_runner as cuda_graph_runner_module

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    ADPShapeAgreement,
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    CUDAGraphRunner,
    cuda_graph_dummy_request_id,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import (
    _classify_dspark_exact_generation_rows,
    _DSPARK_ADP_WIRE_TRAILER_LEN,
    _DSPARK_EXACT_NATIVE_YIELD_INDEX,
    _DSPARK_EXACT_WIRE_PREFIX_LEN,
    _dspark_exact_common_graph_batch_size,
    _dspark_exact_secondary_padding_ready,
    PyExecutor,
    _decode_dspark_exact_expected_yields,
    _validate_dspark_adp_acceptance_gate,
    _validate_dspark_adp_debug_flags,
    _validate_dspark_exact_bucket,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    NoFreeSlotsError,
    ResourceManagerType,
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


def test_cuda_padding_dummy_variant_ids_are_disjoint():
    ids = {
        cuda_graph_dummy_request_id(
            draft_len, variant=variant, max_draft_len=5)
        for variant in (0, 1)
        for draft_len in range(6)
    }
    assert len(ids) == 12
    assert cuda_graph_dummy_request_id(
        5, variant=0,
        max_draft_len=5) == CUDA_GRAPH_DUMMY_REQUEST_ID - 5


def test_exact_common_g_uses_real_peer_counts_and_terminates_all_zero():
    def round_up(rows):
        return 16 if rows <= 16 else 32

    for peer_counts in ([16, 16, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0]):
        payloads = [[0, rows] for rows in peer_counts]
        assert _dspark_exact_common_graph_batch_size(payloads, round_up) == 16

    def reject_round_up(_rows):
        raise AssertionError("all-zero peers must terminate before graph lookup")

    assert _dspark_exact_common_graph_batch_size(
        [[0, 0] for _ in range(8)], reject_round_up) == 0


@pytest.mark.parametrize(
    ("verifier_budget", "zero_peer_ready", "expected"),
    [(48, False, True), (49, False, True), (50, False, False),
     (50, True, True)],
)
def test_secondary_padding_readiness_gates_only_remainders_above_one(
        verifier_budget, zero_peer_ready, expected):
    payloads = [
        [0, 16, 0],
        [0, 0, int(zero_peer_ready)],
    ]
    assert _dspark_exact_secondary_padding_ready(
        payloads,
        graph_batch_size=16,
        verifier_budget=verifier_budget,
        secondary_ready_index=2,
    ) is expected


def _request(*, adp_dummy=False, cuda_dummy=False, generic_dummy=False):
    return SimpleNamespace(
        is_dummy=bool(adp_dummy or cuda_dummy or generic_dummy),
        is_attention_dp_dummy=bool(adp_dummy),
        is_cuda_graph_dummy=bool(cuda_dummy),
        is_dummy_request=bool(generic_dummy),
    )


def test_exact_row_classifier_keeps_idle_adp_dummy_out_of_logical_work():
    real = [_request(), _request()]
    assert _classify_dspark_exact_generation_rows(real) == (real, False)
    adp_dummy = _request(adp_dummy=True)
    assert _classify_dspark_exact_generation_rows([adp_dummy]) == ([], True)
    assert _classify_dspark_exact_generation_rows(
        [_request(), adp_dummy]) == (None, False)
    assert _classify_dspark_exact_generation_rows(
        [_request(cuda_dummy=True)]) == (None, False)


def test_zero_real_nondivisible_padding_uses_distinct_low_high_objects():
    runner = object.__new__(CUDAGraphRunner)
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 4
    runner.config = SimpleNamespace(
        enable_attention_dp=False,
        mapping=SimpleNamespace(tp_size=1),
        batch_size=4,
    )
    runner.adp_shape_agreement = None
    runner.adp_shape_debug = False
    runner.ragged_pad_verify_len = 2
    runner.ragged_zero_real_high_rows = 2
    runner.spec_config = SimpleNamespace(enable_ragged_verify=True,
                                         max_draft_len=5)
    runner._can_run_cuda_graph_batch = lambda _: True
    runner._round_up_batch_size_with_draft_len = lambda *_: 4

    adp = SimpleNamespace(
        is_attention_dp_dummy=True,
        py_verify_len=3,
        py_verify_cap=None,
    )
    low = SimpleNamespace(py_verify_len=None, py_verify_cap=None)
    high = SimpleNamespace(py_verify_len=None, py_verify_cap=None)
    runner._get_or_create_padding_dummy = (
        lambda _resource, _draft, variant=0: low if variant == 0 else high)
    batch = _Batch([adp])

    added = runner._get_padded_batch(
        batch, SimpleNamespace(), runtime_draft_len=5)

    assert added == 3
    assert batch.generation_requests == [adp, high, low, low]
    assert [row.py_verify_len for row in batch.generation_requests] == [
        3, 3, 2, 2
    ]
    assert sum(1 + row.py_verify_len
               for row in batch.generation_requests) == 14


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
    assert payload_len == 23 + exact_cells
    assert debug_index == 22 + exact_cells
    payloads = [[0] * payload_len for _ in range(2)]
    payloads[0][debug_index] = payloads[1][debug_index] = 1
    assert _validate_dspark_adp_debug_flags(payloads, debug_index)


def test_exact_yield_decode_stops_before_four_field_adp_trailer():
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
    # primary/secondary padding-ready, and debug values must not be interpreted
    # as yields.
    trailer_start = _DSPARK_EXACT_WIRE_PREFIX_LEN + len(exact_cells)
    payloads[0][trailer_start:] = [128, 1, 1, 1]
    payloads[1][trailer_start:] = [64, 0, 0, 1]

    native_yield, compact_yields = _decode_dspark_exact_expected_yields(
        payloads=payloads,
        exact_cells=exact_cells,
        graph_batch_size=128,
        yield_scale=1_000_000,
    )

    assert payload_len == 23 + len(exact_cells)
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
        secondary_padding_dummy_requests={},
        ragged_zero_real_high_rows=0,
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
        secondary_padding_dummy_requests={},
        adp_shape_agreement=_agreement(batch),
        _padding_dummy_managers=lambda _resource_manager: [manager],
    )

    assert CUDAGraphRunner.release_padding_dummy(runner, object(), 5)
    assert runner.adp_shape_agreement is None
    assert freed == [dummy]


@pytest.mark.parametrize(
    ("verifier_budget", "scheduled_window", "high_rows"),
    [(80, 4, 0), (81, 5, 1), (88, 5, 8)],
)
def test_zero_real_fit_publishes_only_the_scheduled_dummy_physical_row(
        verifier_budget, scheduled_window, high_rows):
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    runner = SimpleNamespace(
        enabled=True,
        agreed_ragged_bucket=None,
        ragged_pad_verify_len=0,
        ragged_zero_real_high_rows=0,
        supported_batch_sizes=[16],
        secondary_padding_dummy_requests={5: object()},
        _round_up_batch_size=lambda _: 16,
        will_pad_to=lambda *_: True,
    )
    engine = SimpleNamespace(
        cuda_graph_runner=runner,
        spec_config=SimpleNamespace(
            verify_len_tiers=[1, 3, 5], max_draft_len=5),
        _dspark_last_padded_bs=None,
        ragged_verify_token_buckets=lambda _: [48, 64, 80, 81, 88, 96],
        _get_spec_worker=lambda: None,
    )
    adp_dummy = SimpleNamespace(
        is_dummy=True,
        is_attention_dp_dummy=True,
        py_verify_len=None,
    )

    bucket = PyTorchModelEngine.fit_ragged_verify_lens(
        engine,
        [adp_dummy],
        [scheduled_window],
        peer_stats=[[16, 88, 1], [0, 0, 1]],
        exact_shape=(16, verifier_budget, 5),
        exact_zero_real=True,
    )

    assert bucket == verifier_budget
    assert adp_dummy.is_dummy
    assert adp_dummy.py_verify_len == scheduled_window
    assert engine._dspark_last_num_real == 0
    assert runner.ragged_pad_verify_len == 4
    assert runner.ragged_zero_real_high_rows == high_rows


def test_zero_real_fit_rejects_lost_secondary_dummy_after_agreement():
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    runner = SimpleNamespace(
        enabled=True,
        agreed_ragged_bucket=None,
        ragged_pad_verify_len=0,
        ragged_zero_real_high_rows=0,
        supported_batch_sizes=[16],
        secondary_padding_dummy_requests={},
        _round_up_batch_size=lambda _: 16,
        will_pad_to=lambda *_: True,
    )
    engine = SimpleNamespace(
        cuda_graph_runner=runner,
        spec_config=SimpleNamespace(
            verify_len_tiers=[1, 3, 5], max_draft_len=5),
        _dspark_last_padded_bs=None,
        ragged_verify_token_buckets=lambda _: [88],
        _get_spec_worker=lambda: None,
    )
    adp_dummy = SimpleNamespace(
        is_attention_dp_dummy=True, py_verify_len=None)

    with pytest.raises(RuntimeError, match="disappeared after"):
        PyTorchModelEngine.fit_ragged_verify_lens(
            engine,
            [adp_dummy],
            [5],
            peer_stats=[[16, 88, 1], [0, 0, 1]],
            exact_shape=(16, 88, 5),
            exact_zero_real=True,
        )


def test_rebalance_suspends_both_dummy_variants_and_releases_pair_once():
    low = SimpleNamespace(py_request_id=10)
    high = SimpleNamespace(py_request_id=11)
    suspended_ids = []
    resumed_ids = []

    class _Manager:

        def is_request_active(self, request_id):
            return request_id in {10, 11}

        def suspend_request(self, request):
            suspended_ids.append(request.py_request_id)

        def resume_request(self, request):
            resumed_ids.append(request.py_request_id)
            return request.py_request_id != 11

    released = []
    runner = SimpleNamespace(
        padding_dummy_requests={5: low},
        secondary_padding_dummy_requests={5: high},
        release_padding_dummy=lambda resource, draft_len: released.append(
            (resource, draft_len)),
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(cuda_graph_runner=runner)
    executor.resource_manager = object()
    manager = _Manager()

    suspended = executor._suspend_padding_dummies_for_rebalance(manager)
    assert suspended == [(5, 0, low), (5, 1, high)]
    assert suspended_ids == [10, 11]

    executor._resume_padding_dummies_after_rebalance(manager, suspended)
    assert resumed_ids == [10, 11]
    assert released == [(executor.resource_manager, 5)]


def test_padding_dummy_creation_rolls_back_on_spec_slot_failure(monkeypatch):
    dummy = SimpleNamespace(py_request_id=123, is_cuda_graph_dummy=False)
    freed = []

    class _KvManager:

        def add_dummy_requests(self, *args, **kwargs):
            return [dummy]

        def free_resources(self, request):
            freed.append(request)

    class _SpecManager:

        def add_dummy_requests(self, _request_ids):
            raise NoFreeSlotsError("full")

    kv_manager = _KvManager()
    spec_manager = _SpecManager()

    class _Resources:

        def get_resource_manager(self, key):
            if key == "kv":
                return kv_manager
            if key == ResourceManagerType.SPEC_RESOURCE_MANAGER:
                return spec_manager
            return None

    runner = object.__new__(CUDAGraphRunner)
    runner.padding_dummy_requests = {}
    runner.secondary_padding_dummy_requests = {}
    runner.config = SimpleNamespace(
        kv_cache_manager_key="kv",
        use_mrope=False,
        max_beam_width=1,
    )
    runner.spec_config = SimpleNamespace(
        max_draft_len=5,
        get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
    )
    runner.is_encoder_decoder = False
    monkeypatch.setattr(
        cuda_graph_runner_module,
        "get_draft_kv_cache_manager",
        lambda _config, _resources: None,
    )

    assert runner._get_or_create_padding_dummy(
        _Resources(), runtime_draft_len=5, variant=1) is None
    assert freed == [dummy]
    assert runner.secondary_padding_dummy_requests == {}


def test_padding_dummy_rollback_frees_each_registered_manager_once(monkeypatch):
    dummy = SimpleNamespace(py_request_id=456, is_cuda_graph_dummy=False)
    frees = []

    class _Manager:

        def __init__(self, name, *, creates=False):
            self.name = name
            self.creates = creates

        def add_dummy_requests(self, *args, **kwargs):
            if self.name == "spec":
                raise NoFreeSlotsError("full")
            return [dummy] if self.creates else None

        def free_resources(self, _request):
            frees.append(self.name)

    main = _Manager("main", creates=True)
    draft = _Manager("draft")
    cross = _Manager("cross", creates=True)
    spec = _Manager("spec")

    class _Resources:

        def get_resource_manager(self, key):
            return {
                "kv": main,
                ResourceManagerType.CROSS_KV_CACHE_MANAGER: cross,
                ResourceManagerType.SPEC_RESOURCE_MANAGER: spec,
            }.get(key)

    runner = object.__new__(CUDAGraphRunner)
    runner.padding_dummy_requests = {}
    runner.secondary_padding_dummy_requests = {}
    runner.config = SimpleNamespace(
        kv_cache_manager_key="kv",
        use_mrope=False,
        max_beam_width=1,
    )
    runner.spec_config = SimpleNamespace(
        max_draft_len=5,
        get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
    )
    runner.is_encoder_decoder = True
    runner._get_padding_dummy_encoder_output_len = lambda _manager: 1
    monkeypatch.setattr(
        cuda_graph_runner_module,
        "get_draft_kv_cache_manager",
        lambda _config, _resources: draft,
    )

    assert runner._get_or_create_padding_dummy(
        _Resources(), runtime_draft_len=5, variant=1) is None
    assert sorted(frees) == ["cross", "draft", "main"]
    assert len(frees) == len(set(frees))


def test_kv_index_mapper_reserves_two_slots_for_confidence_padding_pair():
    spec_config = SimpleNamespace(enable_confidence_scheduling=True)
    assert KVCacheManagerV2._resolve_num_reserved_index_slots(
        None, spec_config, False) == 2
    assert KVCacheManagerV2._resolve_num_reserved_index_slots(
        None, spec_config, True) == 0
    with pytest.raises(ValueError, match="padding requirement"):
        KVCacheManagerV2._resolve_num_reserved_index_slots(
            1, spec_config, False)
