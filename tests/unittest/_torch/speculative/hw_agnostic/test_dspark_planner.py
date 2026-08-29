# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark planner/budget/cost-table unit tests (hardware-agnostic, CPU).

Covers survival, the budget argmax, cost-table interpolation, top-k
allocation, tier derivation, SGLang decision parity, the verify-length pin,
snapshot instruments, and reconciliation guards.
"""

import hashlib
import itertools
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import (
    ExactSpsCostTable,
    ExactSpsDrainGuard,
    SpsCostTable,
    budget_argmax_over_uniform_lens,
    check_table_fingerprint,
    compute_verify_token_budget,
    derive_verify_len_tiers,
    load_runtime_sps_cost_table,
    load_sps_cost_table,
    load_validated_sps_cost_table,
    select_exact_sps_candidate,
    validate_sps_cost_table_payload,
)
from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)
from tensorrt_llm._torch.speculative.dspark_verify import (
    DSparkVerifyPlanner,
    ExactSpsLocalDecision,
    _exact_cell_geometry,
)

BLOCK = 7


def _drain_guard(
    *,
    tail_graph_batch_size=128,
    loss_multiplier=1.0,
    mean_output_tokens_per_request_iteration=10.0,
    minimum_group_value_ms=0.0,
    source_result_sha256="d" * 64,
):
    return ExactSpsDrainGuard(
        loss_multiplier=loss_multiplier,
        mean_output_tokens_per_request_iteration=(mean_output_tokens_per_request_iteration),
        minimum_group_value_ms=minimum_group_value_ms,
        tail_graph_batch_size=tail_graph_batch_size,
        source_result_sha256=source_result_sha256,
    )


def _cfg(**kw) -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=BLOCK, **kw)


# --------------------------------------------------------------------------
# survival
# --------------------------------------------------------------------------


def test_survival_is_cumulative_product():
    conf = torch.tensor([[0.9, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0]])
    surv = compute_survival(conf)
    assert torch.allclose(surv[0, :3], torch.tensor([0.9, 0.72, 0.36]), atol=1e-6)


# --------------------------------------------------------------------------
# budget argmax
# --------------------------------------------------------------------------


def _brute_force_budget(survival, num_gen, table, min_len=1, max_len=None):
    """Independent O(N) reimplementation used as the oracle for the argmax:
    expected emitted tokens per millisecond over submitted-token cost."""
    bs, blk = survival.shape
    cap = min(max_len or blk, blk)
    cand = np.sort(survival[:, min_len:cap].reshape(-1))[::-1]
    base = num_gen + float(survival[:, :min_len].sum())
    floor_tokens = bs * (min_len + 1)
    best_n, best = 0, -np.inf
    for n in range(cand.size + 1):
        theta = (base + float(cand[:n].sum())) / table.step_time(floor_tokens + n, num_gen)
        if theta > best:
            best, best_n = theta, n
    return best_n


def test_budget_matches_brute_force_on_staircase():
    """Theta is not unimodal on a staircase; a greedy first-descent scan is wrong."""
    rng = np.random.default_rng(7)
    # A cost curve with genuine shelves and risers.
    table = SpsCostTable(
        token_counts=(0, 64, 128, 192, 256, 320),
        step_time_ms=(4.0, 4.05, 6.5, 6.55, 9.0, 9.05),
    )
    for trial in range(25):
        bs = int(rng.integers(1, 40))
        conf = rng.uniform(0.55, 0.99, size=(bs, BLOCK))
        surv = np.cumprod(conf, axis=1)
        got = compute_verify_token_budget(survival=surv, num_gen_requests=bs, cost_table=table)
        want = _brute_force_budget(surv, bs, table)
        assert got == want, f"trial {trial}: bs={bs} got={got} want={want}"


def test_flat_cost_table_degenerates_to_verify_all():
    """A flat cost model makes every token free -> spend the whole budget."""
    table = SpsCostTable.flat(2.0)
    assert table.is_flat
    surv = np.cumprod(np.full((6, BLOCK), 0.9), axis=1)
    budget = compute_verify_token_budget(survival=surv, num_gen_requests=6, cost_table=table)
    assert budget == 6 * (BLOCK - 1)
    # An empty batch has no candidates to buy, whatever the table says.
    assert (
        compute_verify_token_budget(
            survival=np.zeros((0, BLOCK)), num_gen_requests=0, cost_table=table
        )
        == 0
    )


def test_cost_table_interpolates_between_breakpoints():
    """The consumer contract is clamped linear interpolation, not a floor;
    a shelf is only flat if the table measured it as two equal breakpoints."""
    table = SpsCostTable(token_counts=(0, 100), step_time_ms=(1.0, 5.0))
    assert table.step_time(99) > table.step_time(1)
    assert table.step_time(50) == pytest.approx(3.0)
    # A measured shelf stays flat under interpolation.
    shelf = SpsCostTable(token_counts=(0, 99, 100), step_time_ms=(1.0, 1.0, 5.0))
    assert shelf.step_time(1) == shelf.step_time(99)
    # Above the last breakpoint we clamp rather than extrapolate.
    assert table.step_time(10_000) == table.step_time(100)


def test_cost_table_rejects_malformed_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        SpsCostTable(token_counts=(0, 0), step_time_ms=(1.0, 2.0))
    with pytest.raises(ValueError, match="same length"):
        SpsCostTable(token_counts=(0, 1), step_time_ms=(1.0,))
    with pytest.raises(ValueError, match="positive"):
        SpsCostTable(token_counts=(0,), step_time_ms=(0.0,))
    with pytest.raises(ValueError, match="minimum_predicted_gain"):
        SpsCostTable(token_counts=(0,), step_time_ms=(1.0,), minimum_predicted_gain=-0.01)


def test_minimum_gain_guard_falls_back_to_full_budget():
    """A noise-sized compact advantage must preserve ordinary full K."""
    bs, block = 4, 5
    survival = np.ones((bs, block), dtype=np.float64)
    table = SpsCostTable(
        token_counts=(2 * bs, bs * (block + 1)),
        step_time_ms=(3.3, 10.0),
        minimum_predicted_gain=0.02,
    )

    assert compute_verify_token_budget(
        survival=survival,
        num_gen_requests=bs,
        cost_table=table,
        min_verify_len=1,
        max_verify_len=block,
        allowed_lens=[1, block],
    ) == bs * (block - 1)
    assert compute_verify_token_budget(
        survival=survival,
        num_gen_requests=bs,
        cost_table=table,
        min_verify_len=1,
        max_verify_len=block,
    ) == bs * (block - 1)
    assert (
        budget_argmax_over_uniform_lens(
            survival=survival,
            num_gen_requests=bs,
            cost_table=table,
            allowed_lens=[1, block],
        )
        == block
    )


def test_minimum_gain_threshold_edge_keeps_compact_candidate():
    survival = np.ones((1, 2), dtype=np.float64)
    at_threshold = SpsCostTable(
        token_counts=(2, 3),
        step_time_ms=(1.6, 3.0),
        minimum_predicted_gain=0.25,
    )
    above_threshold = SpsCostTable(
        token_counts=(2, 3),
        step_time_ms=(1.6, 3.0),
        minimum_predicted_gain=0.250001,
    )

    assert (
        compute_verify_token_budget(
            survival=survival,
            num_gen_requests=1,
            cost_table=at_threshold,
            allowed_lens=[1, 2],
        )
        == 0
    )
    assert (
        compute_verify_token_budget(
            survival=survival,
            num_gen_requests=1,
            cost_table=at_threshold,
        )
        == 0
    )
    assert (
        budget_argmax_over_uniform_lens(
            survival=survival,
            num_gen_requests=1,
            cost_table=at_threshold,
            allowed_lens=[1, 2],
        )
        == 1
    )
    assert (
        compute_verify_token_budget(
            survival=survival,
            num_gen_requests=1,
            cost_table=above_threshold,
            allowed_lens=[1, 2],
        )
        == 1
    )
    assert (
        budget_argmax_over_uniform_lens(
            survival=survival,
            num_gen_requests=1,
            cost_table=above_threshold,
            allowed_lens=[1, 2],
        )
        == 2
    )


def _multi_g_sps_payload():
    fingerprint = {
        "gpu": "B300",
        "gpu_count": 8,
        "gpu_snapshot_sha256": "a" * 64,
        "global_graph_batch_sizes": [512, 1024],
        "max_draft_len": 5,
        "rank_local_graph_batch_sizes": [64, 128],
        "runtime_snapshot": "runtime-v2",
        "source_diff_sha256": "b" * 64,
        "source_head": "23b73d8",
        "topology": "DEP8",
    }
    cells = {
        64: ((0, 352), (6.0, 5.2)),
        128: ((0, 704, 736), (8.0, 7.0, 7.2)),
    }
    payload = {
        "schema_version": 2,
        "minimum_predicted_gain": 0.02,
        "cost_tables": {
            str(graph_batch_size): {
                "token_counts": list(verifier_budgets),
                "step_time_ms": list(step_times),
            }
            for graph_batch_size, (verifier_budgets, step_times) in cells.items()
        },
        "engine_fingerprint": fingerprint,
        "measurements": [
            {
                "rank_local_graph_batch_size": graph_batch_size,
                "rank_local_verifier_budget": verifier_budget,
                "step_time_ms": step_time,
                "source_result_sha256": "c" * 64,
            }
            for graph_batch_size, (verifier_budgets, step_times) in cells.items()
            for verifier_budget, step_time in zip(verifier_budgets, step_times)
        ],
    }
    payload["engine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _load_test_exact(path, payload):
    return load_validated_sps_cost_table(
        path,
        graph_batch_sizes=[64, 128],
        max_draft_len=5,
        live_engine_fingerprint=dict(payload["engine_fingerprint"]),
    )


def test_exact_cost_table_never_interpolates(tmp_path):
    payload = _multi_g_sps_payload()
    path = tmp_path / "multi-g-sps.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="require load_validated_sps_cost_table"):
        load_sps_cost_table(path)
    table, _ = load_validated_sps_cost_table(
        path,
        graph_batch_sizes=[64, 128],
        max_draft_len=5,
        live_engine_fingerprint=dict(payload["engine_fingerprint"]),
    )

    assert isinstance(table, ExactSpsCostTable)
    assert table.minimum_predicted_gain == pytest.approx(0.02)
    assert table.candidate_budgets(64) == (352,)
    assert table.candidate_budgets(128) == (704, 736)
    assert table.candidate_budgets(128, include_native=True) == (0, 704, 736)
    assert table.step_time(0, 128) == pytest.approx(8.0)
    assert table.step_time(704, 128) == pytest.approx(7.0)
    with pytest.raises(ValueError, match=r"no direct measurements for G=128, V=\[720\]"):
        table.step_time(720, 128)
    with pytest.raises(ValueError, match="no direct measurements for G=256"):
        table.step_time(704, 256)
    with pytest.raises(TypeError, match="requested verifier budget"):
        table.step_time(704.5, 128)


def test_exact_cost_table_rejects_fractional_programmatic_cells():
    table = SpsCostTable(token_counts=(0, 704), step_time_ms=(8.0, 7.0))
    with pytest.raises(TypeError, match="graph batch size"):
        ExactSpsCostTable(tables={128.5: table}, max_draft_len=5)
    with pytest.raises(TypeError, match="measured verifier budget"):
        ExactSpsCostTable(
            tables={128: SpsCostTable(token_counts=(0, 704.5), step_time_ms=(8.0, 7.0))},
            max_draft_len=5,
        )
    with pytest.raises(ValueError, match="V=0 native static K5"):
        ExactSpsCostTable(
            tables={128: SpsCostTable(token_counts=(704,), step_time_ms=(7.0,))},
            max_draft_len=5,
        )


def test_exact_cost_table_identity_covers_grid_costs_and_policy():
    tables = {
        64: SpsCostTable(token_counts=(0, 352), step_time_ms=(5.0, 4.5)),
        128: SpsCostTable(token_counts=(0, 704), step_time_ms=(8.0, 7.0)),
    }
    first = ExactSpsCostTable(tables=tables, max_draft_len=5, minimum_predicted_gain=0.02)
    reordered = ExactSpsCostTable(
        tables=dict(reversed(tuple(tables.items()))),
        max_draft_len=5,
        minimum_predicted_gain=0.02,
    )
    changed_cost = ExactSpsCostTable(
        tables={
            **tables,
            128: SpsCostTable(token_counts=(0, 704), step_time_ms=(8.0, 7.1)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.02,
    )
    changed_policy = ExactSpsCostTable(tables=tables, max_draft_len=5, minimum_predicted_gain=0.03)
    guarded = ExactSpsCostTable(
        tables=tables,
        max_draft_len=5,
        minimum_predicted_gain=0.02,
        iteration_drain_guard=_drain_guard(tail_graph_batch_size=64),
    )

    assert first.identity_sha256 == reordered.identity_sha256
    assert first.collective_identity_words == reordered.collective_identity_words
    assert len(first.collective_identity_words) == 8
    assert all(0 <= value <= 0xFFFFFFFF for value in first.collective_identity_words)
    assert first.identity_sha256 != changed_cost.identity_sha256
    assert first.identity_sha256 != changed_policy.identity_sha256
    assert first.identity_sha256 != guarded.identity_sha256


def test_iteration_drain_guard_rejects_invalid_or_unmeasured_metadata():
    with pytest.raises(ValueError, match="loss_multiplier"):
        _drain_guard(loss_multiplier=0.0)
    with pytest.raises(ValueError, match="mean_output_tokens"):
        _drain_guard(mean_output_tokens_per_request_iteration=float("nan"))
    with pytest.raises(ValueError, match="minimum_group_value_ms"):
        _drain_guard(minimum_group_value_ms=-0.1)

    with pytest.raises(ValueError, match="measured native tail table for G=16"):
        ExactSpsCostTable(
            tables={128: SpsCostTable(token_counts=(0,), step_time_ms=(8.0,))},
            max_draft_len=5,
            iteration_drain_guard=_drain_guard(tail_graph_batch_size=16),
        )


def test_runtime_cost_install_rejects_a_second_different_object():
    planner = DSparkVerifyPlanner(
        cfg=DSparkScheduleConfig(block_size=5, min_verify_len=1),
        cost_table=None,
        tiers=[1, 3, 5],
    )
    authoritative = ExactSpsCostTable(
        tables={128: SpsCostTable(token_counts=(0, 704), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner.install_runtime_cost_table(authoritative)
    planner.install_runtime_cost_table(authoritative)
    assert planner.cost_table is authoritative

    different = ExactSpsCostTable(
        tables={128: SpsCostTable(token_counts=(0, 704), step_time_ms=(8.0, 7.1))},
        max_draft_len=5,
    )
    with pytest.raises(RuntimeError, match="different SPS cost object"):
        planner.install_runtime_cost_table(different)


@pytest.mark.parametrize("invalid_budget", [127, 769])
def test_exact_cost_table_direct_construction_bounds_positive_cells(invalid_budget):
    with pytest.raises(ValueError, match=r"G <= V <= G\*\(K\+1\)"):
        ExactSpsCostTable(
            tables={128: SpsCostTable(token_counts=(0, invalid_budget), step_time_ms=(8.0, 7.0))},
            max_draft_len=5,
        )


def test_legacy_cost_table_loader_remains_interpolating(tmp_path):
    direct = SpsCostTable(token_counts=(0, 100), step_time_ms=(1.0, 5.0))
    assert direct.minimum_predicted_gain == 0.0

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "token_counts": [0, 100],
                "step_time_ms": [1.0, 5.0],
                "_meta": {"lookup": "interp"},
            }
        )
    )

    table, _ = load_sps_cost_table(path)

    assert isinstance(table, SpsCostTable)
    assert table.step_time(50) == pytest.approx(3.0)
    assert table.minimum_predicted_gain == pytest.approx(0.01)

    path.write_text(
        json.dumps(
            {
                "token_counts": [0, 100],
                "step_time_ms": [1.0, 5.0],
                "minimum_predicted_gain": 0.03,
            }
        )
    )
    table, _ = load_sps_cost_table(path)
    assert table.minimum_predicted_gain == pytest.approx(0.03)


def test_exact_cost_table_loader_requires_schema_v2(tmp_path):
    payload = _multi_g_sps_payload()
    payload["schema_version"] = 1
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version=2"):
        load_validated_sps_cost_table(
            path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )

    payload = _multi_g_sps_payload()
    del payload["cost_tables"]
    path = tmp_path / "missing-tables.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required fields: cost_tables"):
        load_validated_sps_cost_table(
            path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )
    payload = _multi_g_sps_payload()
    del payload["schema_version"]
    path = tmp_path / "missing-version.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version=2"):
        load_validated_sps_cost_table(
            path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


def test_runtime_exact_loader_requires_independent_fingerprint_file(tmp_path):
    payload = _multi_g_sps_payload()
    table_path = tmp_path / "exact-sps.json"
    table_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independently generated"):
        load_runtime_sps_cost_table(
            table_path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
        )

    fingerprint_path = tmp_path / "live-fingerprint.json"
    fingerprint_path.write_text(json.dumps(payload["engine_fingerprint"]))
    table, _ = load_runtime_sps_cost_table(
        table_path,
        graph_batch_sizes=[64, 128],
        max_draft_len=5,
        live_engine_fingerprint_path=fingerprint_path,
    )
    assert isinstance(table, ExactSpsCostTable)


@pytest.mark.parametrize("marker", ["engine_fingerprint", "measurements"])
def test_v2_markers_cannot_downgrade_to_legacy(tmp_path, marker):
    exact_payload = _multi_g_sps_payload()
    mixed_payload = {
        "token_counts": [0, 100],
        "step_time_ms": [1.0, 5.0],
        marker: exact_payload[marker],
    }
    path = tmp_path / f"mixed-{marker}.json"
    path.write_text(json.dumps(mixed_payload))

    with pytest.raises(ValueError, match="require load_validated_sps_cost_table"):
        load_sps_cost_table(path)
    with pytest.raises(ValueError, match="schema_version=2"):
        load_validated_sps_cost_table(
            path,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(exact_payload["engine_fingerprint"]),
        )


def test_schema_v2_validation_accepts_exact_runtime_grid():
    payload = _multi_g_sps_payload()

    fingerprint = validate_sps_cost_table_payload(
        payload,
        graph_batch_sizes=[64, 128],
        max_draft_len=5,
        live_engine_fingerprint=dict(payload["engine_fingerprint"]),
    )

    assert fingerprint["source_head"] == "23b73d8"


def test_schema_v2_loads_authenticated_iteration_drain_metadata(tmp_path):
    payload = _multi_g_sps_payload()
    payload["iteration_drain_guard"] = {
        "loss_multiplier": 1.5,
        "mean_output_tokens_per_request_iteration": 4.25,
        "minimum_group_value_ms": 2.0,
        "source_result_sha256": "d" * 64,
        "tail_graph_batch_size": 64,
    }
    path = tmp_path / "multi-g-sps-with-drain-guard.json"
    path.write_text(json.dumps(payload))

    table, _ = _load_test_exact(path, payload)

    assert table.iteration_drain_guard == _drain_guard(
        tail_graph_batch_size=64,
        loss_multiplier=1.5,
        mean_output_tokens_per_request_iteration=4.25,
        minimum_group_value_ms=2.0,
    )


def test_schema_v2_iteration_drain_metadata_requires_measured_tail_g(tmp_path):
    payload = _multi_g_sps_payload()
    payload["iteration_drain_guard"] = {
        "loss_multiplier": 1.5,
        "mean_output_tokens_per_request_iteration": 4.25,
        "minimum_group_value_ms": 2.0,
        "source_result_sha256": "d" * 64,
        "tail_graph_batch_size": 16,
    }
    path = tmp_path / "multi-g-sps-missing-tail-g.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="measured native tail table for G=16"):
        _load_test_exact(path, payload)


def test_schema_v2_validation_requires_the_runtime_graph_grid():
    payload = _multi_g_sps_payload()
    with pytest.raises(ValueError, match="must match exactly"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


def test_schema_v2_validation_requires_independent_live_fingerprint():
    payload = _multi_g_sps_payload()
    with pytest.raises(ValueError, match="independently supplied"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
        )
    with pytest.raises(ValueError, match="independently of the SPS"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=payload["engine_fingerprint"],
        )


def test_schema_v2_validation_authenticates_and_matches_fingerprint():
    payload = _multi_g_sps_payload()
    live = dict(payload["engine_fingerprint"])
    payload["engine_fingerprint"]["topology"] = "TEP8"
    with pytest.raises(ValueError, match="SHA256"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=live,
        )

    payload = _multi_g_sps_payload()
    live = dict(payload["engine_fingerprint"])
    live["runtime_snapshot"] = "different"
    with pytest.raises(ValueError, match="does not match active runtime"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=live,
        )


def test_schema_v2_validation_requires_native_v0_for_every_g():
    payload = _multi_g_sps_payload()
    payload["cost_tables"]["128"]["token_counts"].pop(0)
    payload["cost_tables"]["128"]["step_time_ms"].pop(0)
    payload["measurements"] = [
        item
        for item in payload["measurements"]
        if not (
            item["rank_local_graph_batch_size"] == 128 and item["rank_local_verifier_budget"] == 0
        )
    ]

    with pytest.raises(ValueError, match="V=0 native static K5"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


@pytest.mark.parametrize("invalid_budget", [64, 784])
def test_schema_v2_validation_bounds_positive_verifier_budgets(invalid_budget):
    payload = _multi_g_sps_payload()
    original_budget = 704 if invalid_budget == 64 else 736
    table_index = 1 if invalid_budget == 64 else 2
    payload["cost_tables"]["128"]["token_counts"][table_index] = invalid_budget
    for measurement in payload["measurements"]:
        if (
            measurement["rank_local_graph_batch_size"] == 128
            and measurement["rank_local_verifier_budget"] == original_budget
        ):
            measurement["rank_local_verifier_budget"] = invalid_budget

    with pytest.raises(ValueError, match=r"G <= V <= G\*\(K\+1\)"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


def test_schema_v2_validation_rejects_extra_or_mismatched_provenance():
    payload = _multi_g_sps_payload()
    payload["cost_tables"]["128"]["token_counts"].append(752)
    payload["cost_tables"]["128"]["step_time_ms"].append(7.4)
    with pytest.raises(ValueError, match="cells must match exactly"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )

    payload = _multi_g_sps_payload()
    payload["measurements"][0]["step_time_ms"] += 0.1
    with pytest.raises(ValueError, match="do not match measurement"):
        validate_sps_cost_table_payload(
            payload,
            graph_batch_sizes=[64, 128],
            max_draft_len=5,
            live_engine_fingerprint=dict(payload["engine_fingerprint"]),
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda payload: payload.update({"unknown": 1}), "unknown fields"),
        (lambda payload: payload.update({"measurements": None}), "null fields"),
        (lambda payload: payload["cost_tables"]["128"].update({"unknown": 1}), "unknown fields"),
        (lambda payload: payload["engine_fingerprint"].update({"unknown": 1}), "unknown fields"),
    ],
)
def test_schema_v2_rejects_null_and_unknown_fields(tmp_path, mutation, message):
    payload = _multi_g_sps_payload()
    mutation(payload)
    path = tmp_path / "invalid-shape.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        _load_test_exact(path, payload)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda payload: payload["cost_tables"]["128"]["token_counts"].__setitem__(1, 704.0),
            "JSON integer",
        ),
        (
            lambda payload: payload["measurements"][1].update(
                {"rank_local_verifier_budget": 352.5}
            ),
            "JSON integer",
        ),
        (
            lambda payload: payload.update({"minimum_predicted_gain": -0.1}),
            "non-negative and finite",
        ),
        (
            lambda payload: payload.update({"minimum_predicted_gain": float("nan")}),
            "non-negative and finite",
        ),
        (
            lambda payload: payload["cost_tables"]["64"]["step_time_ms"].__setitem__(0, 0.0),
            "positive and finite",
        ),
        (
            lambda payload: payload["cost_tables"]["64"]["step_time_ms"].__setitem__(
                0, float("inf")
            ),
            "positive and finite",
        ),
    ],
)
def test_schema_v2_rejects_fractional_or_invalid_numbers(tmp_path, mutation, message):
    payload = _multi_g_sps_payload()
    mutation(payload)
    path = tmp_path / "invalid-number.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match=message):
        _load_test_exact(path, payload)


def test_schema_v2_rejects_fractional_graph_key(tmp_path):
    payload = _multi_g_sps_payload()
    payload["cost_tables"]["128.0"] = payload["cost_tables"].pop("128")
    path = tmp_path / "fractional-g.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="canonical positive integer string"):
        _load_test_exact(path, payload)


def test_exact_selector_keeps_native_on_tie_or_below_threshold():
    table = ExactSpsCostTable(
        tables={128: SpsCostTable(token_counts=(0, 704), step_time_ms=(10.0, 8.0))},
        max_draft_len=5,
        minimum_predicted_gain=0.02,
        iteration_drain_guard=_drain_guard(),
    )

    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.0},
            cost_table=table,
        )
        == 0
    )
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.08},
            cost_table=table,
        )
        == 0
    )
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={704: 8.24},
            cost_table=table,
        )
        == 704
    )
    with pytest.raises(ValueError, match="must not include native V=0"):
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={0: 10.0, 704: 8.24},
            cost_table=table,
        )


def test_exact_selector_fails_closed_without_iteration_drain_metadata():
    table = ExactSpsCostTable(
        tables={128: SpsCostTable(token_counts=(0, 512), step_time_ms=(100.0, 80.0))},
        max_draft_len=5,
        minimum_predicted_gain=0.0,
    )

    # Compact has positive immediate expected goodput, but the table cannot
    # price its workload-specific iteration loss.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={512: 8.5},
            cost_table=table,
        )
        == 0
    )


def test_exact_selector_applies_strict_iteration_drain_group_value():
    table = ExactSpsCostTable(
        tables={
            16: SpsCostTable(token_counts=(0,), step_time_ms=(40.0,)),
            128: SpsCostTable(token_counts=(0, 512), step_time_ms=(100.0, 80.0)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.0,
        iteration_drain_guard=_drain_guard(
            tail_graph_batch_size=16,
            loss_multiplier=1.5,
            mean_output_tokens_per_request_iteration=5.0,
            minimum_group_value_ms=2.0,
        ),
    )

    # T(128,0)-T(128,512) - 1.5*(10-8.5)*T(16,0)/5 == 2ms.
    # Equality is deliberately native.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={512: 8.5},
            cost_table=table,
        )
        == 0
    )
    # A 1.4-token predicted loss has group value 3.2ms and is admitted.
    assert (
        select_exact_sps_candidate(
            graph_batch_size=128,
            native_expected_yield=10.0,
            compact_expected_yields={512: 8.6},
            cost_table=table,
        )
        == 512
    )


@pytest.mark.parametrize("graph_batch_size", [16, 32, 64])
def test_exact_selector_fails_closed_without_measured_compact_cell(graph_batch_size):
    table = ExactSpsCostTable(
        tables={
            16: SpsCostTable(token_counts=(0,), step_time_ms=(40.0,)),
            32: SpsCostTable(token_counts=(0,), step_time_ms=(55.0,)),
            64: SpsCostTable(token_counts=(0,), step_time_ms=(70.0,)),
            128: SpsCostTable(token_counts=(0, 512), step_time_ms=(100.0, 80.0)),
        },
        max_draft_len=5,
        minimum_predicted_gain=0.0,
        iteration_drain_guard=_drain_guard(tail_graph_batch_size=16),
    )

    assert (
        select_exact_sps_candidate(
            graph_batch_size=graph_batch_size,
            native_expected_yield=10.0,
            compact_expected_yields={},
            cost_table=table,
        )
        == 0
    )


def test_exact_production_candidates_exclude_full_ragged_control():
    table = ExactSpsCostTable(
        tables={128: SpsCostTable(token_counts=(0, 704, 768), step_time_ms=(8.0, 7.0, 8.1))},
        max_draft_len=5,
    )
    assert table.candidate_budgets(128) == (704, 768)
    assert table.production_candidate_budgets(128) == (704,)
    assert table.candidate_cells() == ((128, 704),)


def test_exact_table_limits_compact_graph_cells_per_g():
    with pytest.raises(ValueError, match="compact V cells per G"):
        ExactSpsCostTable(
            tables={
                128: SpsCostTable(
                    token_counts=(0, 128, 256, 384, 512, 640),
                    step_time_ms=(8.0, 7.9, 7.8, 7.7, 7.6, 7.5),
                )
            },
            max_draft_len=5,
        )


def test_exact_cell_geometry_models_shared_pad_window():
    # 100 real rows at K5 can absorb 600 tokens. V704 at G128 therefore
    # requires every one of the 28 pad rows to carry four tokens.
    assert _exact_cell_geometry(
        num_real=100,
        graph_batch_size=128,
        verifier_budget=704,
        min_verify_len=1,
        max_verify_len=5,
    ) == (4, 592, 392)
    # V704 = 128*5 + 64: the idle rank uses 64 six-token rows and 64
    # five-token rows without assigning any logical request work.
    assert _exact_cell_geometry(
        num_real=0,
        graph_batch_size=128,
        verifier_budget=704,
        min_verify_len=1,
        max_verify_len=5,
    ) == (5, 0, 0)


def test_exact_zero_real_rank_advertises_pad_only_cells_without_yield():
    table = ExactSpsCostTable(
        tables={
            4: SpsCostTable(
                token_counts=(0, 12, 14, 24),
                step_time_ms=(8.0, 6.0, 6.5, 8.1),
            )
        },
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(
        cfg=DSparkScheduleConfig(block_size=5, min_verify_len=1),
        cost_table=table,
        tiers=[1, 3, 5],
        detailed_stats=True,
    )

    decision = planner.prepare_exact_sps_decision(
        num_gen_requests=0, rows=[])

    assert decision is not None
    assert decision.num_requests == 0
    assert tuple(decision.survival.shape) == (0, 5)
    assert decision.native_expected_yield == 0.0
    assert decision.compact_expected_yields == (0.0, 0.0)
    lens, budget, pad_tokens = planner.allocate_exact_sps_candidate(
        decision, graph_batch_size=4, verifier_budget=14)
    assert lens == []
    assert budget == 0
    assert pad_tokens == 3
    assert planner.stats["exact_cell_hist"] == {"G4V14": 1}
    assert planner.stats["exact_zero_real_ready_steps"] == 1
    assert planner.stats["exact_zero_real_selected_steps"] == 1
    assert planner.stats["exact_zero_real_dummy_row_split_hist"] == {
        "q3r2G4": 1
    }


def test_exact_allocator_spends_the_modeled_real_target():
    table = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 22), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(
        cfg=DSparkScheduleConfig(block_size=5, min_verify_len=1),
        cost_table=table,
        tiers=[1, 3, 5],
    )
    decision = ExactSpsLocalDecision(
        num_requests=3,
        survival=torch.ones((3, 5)),
        native_expected_yield=18.0,
        compact_expected_yields=(18.0,),
    )
    lens, budget, pad_tokens = planner.allocate_exact_sps_candidate(
        decision, graph_batch_size=4, verifier_budget=22
    )
    assert lens == [5, 5, 5]
    assert budget == 12
    assert pad_tokens == 4
    assert sum(value + 1 for value in lens) + pad_tokens == 22
    assert "predicted_steps" not in planner.stats
    assert "predicted_ms_sum" not in planner.stats
    assert "exact_cell_hist" not in planner.stats
    assert not hasattr(planner, "last_predicted_step_ms")


def test_exact_planner_does_not_record_fallbacks_without_detailed_stats():
    table = ExactSpsCostTable(
        tables={4: SpsCostTable(token_counts=(0, 12), step_time_ms=(8.0, 7.0))},
        max_draft_len=5,
    )
    planner = DSparkVerifyPlanner(
        cfg=DSparkScheduleConfig(block_size=5, min_verify_len=1),
        cost_table=table,
        tiers=[1, 3, 5],
    )

    assert planner.prepare_exact_sps_decision(num_gen_requests=-1) is None
    assert planner.prepare_exact_sps_decision(num_gen_requests=1) is None
    assert planner.stats == {
        "decisions": 0,
        "fallback_no_snapshot": 0,
        "fallback_flat_cost": 0,
        "fallback_no_confidence": 0,
        "fallback_short_snapshot": 0,
        "fallback_no_gen_requests": 0,
        "fallback_full_k": 0,
        "forced_steps": 0,
    }


# --------------------------------------------------------------------------
# per-request allocation
# --------------------------------------------------------------------------


def test_allocation_is_a_prefix_and_respects_bounds():
    torch.manual_seed(3)
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.rand(12, BLOCK))
    lens = schedule_verify_lens_topk(survival=surv, budget=20, cfg=cfg)
    assert lens.dtype == torch.int32
    assert torch.all(lens >= cfg.min_verify_len)
    assert torch.all(lens <= cfg.resolved_max_verify_len)
    # Budget is spent above the floor and never exceeded.
    assert int((lens - cfg.min_verify_len).sum()) <= 20
    # Zero budget degenerates to the floor for everyone.
    cfg_floor = _cfg(min_verify_len=2)
    lens = schedule_verify_lens_topk(
        survival=compute_survival(torch.rand(5, BLOCK)), budget=0, cfg=cfg_floor
    )
    assert torch.equal(lens, torch.full((5,), 2, dtype=torch.int32))
    # A huge budget saturates at the cap.
    cfg_cap = _cfg(min_verify_len=1, max_verify_len=4)
    lens = schedule_verify_lens_topk(
        survival=compute_survival(torch.full((5, BLOCK), 0.99)), budget=10**6, cfg=cfg_cap
    )
    assert torch.equal(lens, torch.full((5,), 4, dtype=torch.int32))
    # An empty batch allocates nothing.
    assert (
        schedule_verify_lens_topk(survival=torch.zeros(0, BLOCK), budget=5, cfg=_cfg()).numel() == 0
    )


def test_survival_eps_excludes_hopeless_positions():
    """Even with unlimited budget, sub-eps candidates are never admitted."""
    cfg = _cfg(min_verify_len=1, survival_eps=1e-3)
    conf = torch.tensor([[1.0, 1.0, 1e-9, 1.0, 1.0, 1.0, 1.0]])
    lens = schedule_verify_lens_topk(survival=compute_survival(conf), budget=10**6, cfg=cfg)
    # positions 0,1 survive; from position 2 on, survival collapses below eps.
    assert int(lens[0]) == 2


def test_allocation_is_deterministic_under_tied_confidences():
    """Two ranks with identical input must produce identical verify lengths."""
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.full((9, BLOCK), 0.9))
    first = schedule_verify_lens_topk(survival=surv, budget=13, cfg=cfg)
    for _ in range(8):
        assert torch.equal(schedule_verify_lens_topk(survival=surv, budget=13, cfg=cfg), first)


def test_ties_break_toward_earlier_positions():
    """All candidates tie, so the allocation must stay a balanced prefix front,
    not hand one request a deep suffix while another gets nothing."""
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.full((4, BLOCK), 1.0))
    lens = schedule_verify_lens_topk(survival=surv, budget=4, cfg=cfg)
    assert int(lens.max()) - int(lens.min()) <= 1


def test_schedule_config_rejects_bad_bounds():
    with pytest.raises(ValueError, match="min_verify_len must be >= 1"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=0)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DSparkScheduleConfig(block_size=0)
    with pytest.raises(ValueError, match="< min_verify_len"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=5, max_verify_len=3)
    with pytest.raises(ValueError, match=r"\[bs, K\]"):
        compute_survival(torch.rand(BLOCK))


# --------------------------------------------------------------------------
# budget conservation: allocation must never exceed what the planner sized
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bs,min_len", list(itertools.product([1, 3, 17], [1, 2])))
def test_planner_and_allocator_agree_on_the_budget(bs, min_len):
    rng = np.random.default_rng(bs * 100 + min_len)
    table = SpsCostTable(token_counts=(0, 16, 32, 64, 128), step_time_ms=(3.0, 3.1, 5.0, 5.1, 9.0))
    conf = rng.uniform(0.4, 0.99, size=(bs, BLOCK))
    surv_np = np.cumprod(conf, axis=1)
    budget = compute_verify_token_budget(
        survival=surv_np, num_gen_requests=bs, cost_table=table, min_verify_len=min_len
    )
    lens = schedule_verify_lens_topk(
        survival=torch.from_numpy(surv_np).float(),
        budget=budget,
        cfg=_cfg(min_verify_len=min_len),
    )
    assert int((lens - min_len).sum()) <= budget
    assert torch.all(lens >= min_len)


# --------------------------------------------------------------------------
# uniform-K projection (what TRT-LLM's graph key can actually express today)
# --------------------------------------------------------------------------


def test_discrete_argmax_beats_round_then_snap_across_a_riser():
    """Optimizing over runnable lengths is not the same as rounding the budget:
    snapping down past a cost riser lands on a worse Theta than the search."""
    table = SpsCostTable(token_counts=(0, 8, 24, 40), step_time_ms=(2.0, 2.05, 2.1, 20.0))
    surv = np.cumprod(np.full((8, BLOCK), 0.97), axis=1)
    chosen = budget_argmax_over_uniform_lens(
        survival=surv,
        num_gen_requests=8,
        cost_table=table,
        allowed_lens=[1, 3, 5, 7],
    )
    # 8*(7+1)=64 tokens lands on the 20ms riser; the search must avoid it.
    assert chosen < 7

    def theta(length):
        # Mirrors the implementation exactly, including the bonus token every
        # request submits alongside its drafts.
        tau = 8 + float(surv[:, :length].sum())
        return tau / table.step_time(8 * (length + 1), 8)

    assert theta(chosen) == max(theta(v) for v in (1, 3, 5, 7))
    # With nothing runnable, fall back to the floor.
    assert (
        budget_argmax_over_uniform_lens(
            survival=surv, num_gen_requests=8, cost_table=table, allowed_lens=[]
        )
        == 1
    )


def test_derived_tiers_lose_nothing_versus_continuous_k():
    """The shelf-right-edge property: within a shelf Theta rises monotonically,
    so tiers built from right edges must match an exhaustive length search."""
    rng = np.random.default_rng(11)
    # Shelves are encoded as breakpoint pairs with equal values: under the
    # interpolating consumer, flatness between points is only real when the
    # table measured it.
    table = SpsCostTable(
        token_counts=(0, 11, 12, 29, 30, 47, 48, 89, 90, 159, 160),
        step_time_ms=(2.0, 2.0, 2.4, 2.4, 3.9, 3.9, 4.0, 4.0, 7.5, 7.5, 12.0),
    )
    for bs in (3, 8, 16, 31):
        surv = np.cumprod(rng.uniform(0.5, 0.995, size=(bs, BLOCK)), axis=1)
        tiers = derive_verify_len_tiers(
            cost_table=table, num_requests=bs, block_size=BLOCK, max_tiers=99
        )

        def theta(length):
            return (bs + float(surv[:, :length].sum())) / table.step_time(bs * (length + 1), bs)

        best_any = max(theta(v) for v in range(1, BLOCK + 1))
        best_tier = max(theta(v) for v in tiers)
        assert best_tier == pytest.approx(best_any), f"bs={bs} tiers={tiers}"


def test_derived_tiers_respect_the_capture_budget():
    table = SpsCostTable(
        token_counts=(0, 5, 9, 14, 20, 27, 35), step_time_ms=(1.0, 2, 3, 4, 5, 6, 7)
    )
    tiers = derive_verify_len_tiers(cost_table=table, num_requests=2, block_size=BLOCK, max_tiers=3)
    assert len(tiers) <= 3
    # Endpoints are always kept: the floor is always runnable and the full block
    # is always a right edge (the last shelf is unbounded).
    assert tiers[0] == 1 and tiers[-1] == BLOCK


def test_derived_tiers_are_a_function_of_batch_size():
    """Tiers move with bs because total tokens = bs * (length + 1); a tier set
    derived once and reused across batch sizes would be wrong for one of them."""
    table = SpsCostTable(token_counts=(0, 40), step_time_ms=(1.0, 5.0))
    assert derive_verify_len_tiers(
        cost_table=table, num_requests=8, block_size=BLOCK, max_tiers=99
    ) == [1, 3, BLOCK]
    assert derive_verify_len_tiers(
        cost_table=table, num_requests=32, block_size=BLOCK, max_tiers=99
    ) == [1, BLOCK]


# --------------------------------------------------------------------------
# host-side planner: relay, fallbacks, cross-rank agreement
# --------------------------------------------------------------------------


def _planner(**kw):
    kw.setdefault("cfg", _cfg(min_verify_len=1))
    kw.setdefault(
        "cost_table",
        SpsCostTable(token_counts=(0, 8, 24, 48), step_time_ms=(2.0, 2.1, 4.0, 9.0)),
    )
    kw.setdefault("tiers", [1, 3, BLOCK])
    return DSparkVerifyPlanner(**kw)


def test_planner_reads_confidence_by_row_not_by_batch_position():
    """The snapshot is slot-indexed and read one iteration later; ``rows`` is
    what re-associates it after joins and departures reshuffle the batch."""
    p = _planner()
    # Row 3 is a confident request, row 0 a hopeless one.
    buf = torch.full((5, BLOCK), -8.0)
    buf[3] = 8.0
    p._host_buffer, p._copy_event, p._snapshot_valid = buf, None, True

    # This step schedules the confident request first, then the hopeless one.
    lens = p.decide_verify_lens(num_gen_requests=2, rows=[3, 0])
    assert lens is not None and lens[0] > lens[1], f"row mapping ignored: {lens}"


def test_planner_rejects_a_row_list_that_does_not_match_the_batch():
    p = _planner()
    p._host_buffer, p._copy_event, p._snapshot_valid = torch.rand(8, BLOCK), None, True
    assert p.decide_verify_lens(num_gen_requests=4, rows=[0, 1]) is None
    assert p.stats["fallback_short_snapshot"] == 1


def test_planner_routes_a_selected_full_budget_to_native_static_k():
    """Full K is a fallback decision, not a reason to pack a ragged graph."""
    bs, block = 4, 5
    planner = DSparkVerifyPlanner(
        cfg=DSparkScheduleConfig(block_size=block, min_verify_len=1),
        cost_table=SpsCostTable(
            token_counts=(0, bs * (block + 1)),
            step_time_ms=(10.0, 10.01),
            minimum_predicted_gain=0.01,
        ),
        tiers=[1, 3, block],
    )
    planner._gather_rows = lambda **_: torch.full((bs, block), 8.0)
    assert planner.decide_verify_lens(num_gen_requests=bs, reduce_across_ranks=False) is None
    assert planner.stats["fallback_full_k"] == 1


def test_cross_rank_full_bucket_routes_to_native_static_k():
    """A compact local choice promoted to full GxK must skip ragged work."""
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    planner = types.SimpleNamespace(
        forced_verify_len=None,
        forced_budget_frac=None,
        detailed_stats=True,
        stats={},
    )
    runner = types.SimpleNamespace(
        enabled=True,
        agreed_ragged_bucket=999,
        ragged_zero_real_high_rows=0,
        supported_batch_sizes=[4],
        _round_up_batch_size=lambda _: 4,
        will_pad_to=lambda *_: True,
    )
    engine = types.SimpleNamespace(
        cuda_graph_runner=runner,
        spec_config=types.SimpleNamespace(verify_len_tiers=[1, 3, 5]),
        _dspark_last_padded_bs=999,
        ragged_verify_token_buckets=lambda _: [8, 16, 24],
        _get_spec_worker=lambda: types.SimpleNamespace(verify_planner=planner),
    )

    bucket = PyTorchModelEngine.fit_ragged_verify_lens(engine, [object()] * 4, [5] * 4)
    assert bucket is None
    assert runner.agreed_ragged_bucket is None
    assert engine._dspark_last_padded_bs is None
    assert planner.stats["fallback_full_k"] == 1


def test_exact_fit_executes_the_selected_bucket_without_rounding(monkeypatch):
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
    from tensorrt_llm._torch.speculative.dspark_ragged import RaggedVerifyLayout

    def reject_repacking(*_args, **_kwargs):
        raise AssertionError("exact fit must publish the selected layout directly")

    monkeypatch.setattr(RaggedVerifyLayout, "from_verify_lens", reject_repacking)

    runner = types.SimpleNamespace(
        enabled=True,
        agreed_ragged_bucket=None,
        ragged_pad_verify_len=0,
        ragged_zero_real_high_rows=0,
        supported_batch_sizes=[4],
        _round_up_batch_size=lambda _: 4,
        will_pad_to=lambda *_: True,
    )
    engine = types.SimpleNamespace(
        cuda_graph_runner=runner,
        spec_config=types.SimpleNamespace(verify_len_tiers=[1, 3, 5]),
        _dspark_last_padded_bs=None,
        ragged_verify_token_buckets=lambda _: [22],
        _get_spec_worker=lambda: None,
    )
    requests = [types.SimpleNamespace() for _ in range(3)]
    bucket = PyTorchModelEngine.fit_ragged_verify_lens(
        engine,
        requests,
        [5, 5, 5],
        peer_stats=[[3, 18, 1]],
        exact_shape=(4, 22, 4),
    )
    assert bucket == 22
    assert runner.agreed_ragged_bucket == 22
    assert runner.ragged_pad_verify_len == 3
    assert [request.py_verify_len for request in requests] == [5, 5, 5]


def test_static_warmup_observability_is_safe_without_spec_worker():
    """Static warmup may record graph use before a DSpark worker exists."""
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    engine = object.__new__(PyTorchModelEngine)
    engine.model = types.SimpleNamespace()

    assert engine._get_spec_worker() is None
    assert engine._dspark_ragged_stats is None
    engine._record_dspark_graph_use(replayed=False, shape="warmup")


def test_feature_off_metadata_paths_do_not_scan_request_windows():
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    class RequestWithoutWindowAccess:
        @property
        def py_verify_len(self):
            raise AssertionError("feature-off path scanned request windows")

        @property
        def py_verify_cap(self):
            raise AssertionError("feature-off path scanned request caps")

    requests = [RequestWithoutWindowAccess()]
    attention = types.SimpleNamespace(ragged_verify_lens=object())
    engine = types.SimpleNamespace(
        _dspark_computes_windows=False,
        _dspark_device_windows=False,
        _dspark_trims_submitted_tokens=False,
    )
    PyTorchModelEngine._publish_gen_token_layout(engine, attention, requests)
    assert attention.ragged_verify_lens is None

    speculative = types.SimpleNamespace(
        accept_caps=object(),
        accept_cap_trim=object(),
        qo_indptr=object(),
        total_verify_tokens=1,
        verify_lens=object(),
    )
    PyTorchModelEngine._attach_ragged_verify_layout(engine, speculative, attention, requests)
    assert speculative.accept_caps is None
    assert speculative.accept_cap_trim is None
    assert speculative.verify_lens is None
    assert speculative.qo_indptr is None
    assert speculative.total_verify_tokens is None


def _capture_set(batch_sizes, tiers, max_draft_len=BLOCK, exact_table=None):
    """Invoke PyTorchModelEngine._get_graphs_to_capture with a stub engine."""
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    spec_config = types.SimpleNamespace(
        enable_confidence_scheduling=True,
        verify_len_tiers=list(tiers),
        draft_len_schedule=None,
        max_concurrency=None,
        is_linear_tree=True,
        spec_dec_mode=types.SimpleNamespace(
            support_dynamic_draft_len=lambda: False, use_one_engine=lambda: False
        ),
    )
    engine = types.SimpleNamespace(
        is_draft_model=False,
        spec_config=spec_config,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_draft_len,
        original_max_draft_len=max_draft_len,
        _dynamic_draft_len_mapping=None,
        _cuda_graph_batch_sizes=list(batch_sizes),
        _dspark_sps_cost_table=exact_table,
    )
    return PyTorchModelEngine._get_graphs_to_capture(engine, list(batch_sizes))


def test_capture_pins_the_draft_length_to_the_top_tier():
    """One graph per batch size, not a cross product with the tier ladder:
    nothing varies draft_len at runtime, so lower-tier graphs are unreachable."""
    bss, tiers = [1, 8, 64], [1, 3, BLOCK]
    graphs = _capture_set(bss, tiers)
    assert set(graphs) == {(bs, BLOCK) for bs in bss}
    assert len(graphs) == len(bss), (
        f"expected one graph per batch size, got {len(graphs)}: {graphs}"
    )


def test_compact_capture_preserves_native_static_graph_for_every_g():
    """Full-K policy fallbacks must replay the same native keys as static K5."""
    bss, tiers = [16, 32, 64, 128], [1, 3, 5]
    with patch.dict(os.environ, {"TLLM_DSPARK_RAGGED_VERIFY_MODE": "compact"}):
        graphs = _capture_set(bss, tiers, max_draft_len=5)

    native = {(bs, 5) for bs in bss}
    ragged = {(bs, 5, bs * (tier + 1)) for bs in bss for tier in tiers}
    assert set(graphs) == native | ragged
    assert len(graphs) == len(native) + len(ragged)


def test_exact_compact_capture_uses_only_measured_production_cells():
    """Exact mode captures native fallback plus profitable measured V cells."""
    bss = [16, 32]
    table = ExactSpsCostTable(
        tables={
            16: SpsCostTable(token_counts=(0, 80, 96), step_time_ms=(2.0, 1.8, 2.1)),
            32: SpsCostTable(token_counts=(0, 160, 192), step_time_ms=(3.0, 2.7, 3.1)),
        },
        max_draft_len=5,
    )
    with patch.dict(os.environ, {"TLLM_DSPARK_RAGGED_VERIFY_MODE": "compact"}):
        graphs = _capture_set(bss, [1, 3, 5], max_draft_len=5, exact_table=table)

    assert set(graphs) == {
        (16, 5),
        (32, 5),
        (16, 5, 80),
        (32, 5, 160),
    }


# --------------------------------------------------------------------------
# top-k allocation follows confidence
# --------------------------------------------------------------------------


def _block5_cfg() -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)


def test_topk_hands_longer_windows_to_more_confident_requests():
    """The budget must follow survival, not batch position."""
    # survival[r, k] = P(the first k+1 drafted tokens all get accepted)
    survival = torch.tensor(
        [
            [0.99, 0.98, 0.97, 0.96, 0.95],  # confident
            [0.90, 0.70, 0.50, 0.30, 0.10],  # middling
            [0.20, 0.04, 0.01, 0.00, 0.00],  # collapses
        ]
    )
    lens = schedule_verify_lens_topk(survival=survival, budget=6, cfg=_block5_cfg()).tolist()

    assert len(lens) == 3
    assert lens[0] > lens[2], (
        f"confident request got {lens[0]}, collapsing request got {lens[2]}; "
        f"the budget must follow survival, not batch position"
    )
    assert lens[0] >= lens[1] >= lens[2]
    # Every request keeps at least the floor, and the budget is respected.
    assert min(lens) >= 1
    assert sum(lens) - len(lens) * 1 <= 6


def test_a_full_budget_degenerates_to_the_uniform_full_window():
    """The no-trim case must stay exactly uniform: ragged must cost nothing."""
    num_reqs, max_len = 4, 5
    survival = torch.full((num_reqs, max_len), 0.99)
    budget = num_reqs * (max_len - 1)
    lens = schedule_verify_lens_topk(survival=survival, budget=budget, cfg=_block5_cfg()).tolist()
    assert lens == [max_len] * num_reqs


# --------------------------------------------------------------------------
# cost-table interpolation: clamped linear on both the token and batch axes
# --------------------------------------------------------------------------

# The certified GB300 table's shape, abbreviated to two breakpoints.
TABLE = SpsCostTable(
    token_counts=(512, 768, 1536),
    step_time_ms=(23.85, 35.61, 105.64),
    fixed_overhead_ms=25.244,
    batch_sizes=(128, 256),
    batch_overhead_ms=(11.462, 19.343),
)


def test_theta_interpolates_between_breakpoints():
    """Exact on breakpoints, linear between them: 1512 tokens must cost ~the
    1536 price, not the 768 one a floor lookup would return."""
    assert TABLE.step_time(768, 256) == pytest.approx(25.244 + 19.343 + 35.61)
    assert TABLE.step_time(1536, 256) == pytest.approx(25.244 + 19.343 + 105.64)
    got = TABLE.step_time(1512, 252)
    theta = 35.61 + (105.64 - 35.61) * (1512 - 768) / (1536 - 768)
    alpha = 11.462 + (19.343 - 11.462) * (252 - 128) / (256 - 128)
    assert got == pytest.approx(25.244 + alpha + theta)
    # The floor price it used to return -- and must never return again.
    assert got > 130.0


def test_alpha_interpolates_and_clamps():
    assert TABLE.batch_overhead(128) == pytest.approx(11.462)
    assert TABLE.batch_overhead(192) == pytest.approx((11.462 + 19.343) / 2)
    assert TABLE.batch_overhead(64) == pytest.approx(11.462)  # clamp low
    assert TABLE.batch_overhead(512) == pytest.approx(19.343)  # clamp high


def test_theta_clamps_outside_the_measured_range():
    """Theta clamps to the end values; a table without a batch axis adds no alpha."""
    plain = SpsCostTable(
        token_counts=(512, 768, 1536), step_time_ms=(23.85, 35.61, 105.64), fixed_overhead_ms=25.244
    )
    assert plain.step_time(100, 0) == pytest.approx(25.244 + 23.85)
    assert plain.step_time(4096, 0) == pytest.approx(25.244 + 105.64)


# --------------------------------------------------------------------------
# decision-level parity with the SGLang budget planner this was ported from
# --------------------------------------------------------------------------

# Reference implementation, copied from SGLang dspark_components/dspark_planner
# (additive-table path) with only cosmetic renames. Do not "fix" this copy:
# its job is to be what SGLang runs.


def _sgl_interp_clamped(xs, ys, x: float) -> float:
    xs = torch.tensor(xs, dtype=torch.float64)
    ys = torch.tensor(ys, dtype=torch.float64)
    x_t = torch.tensor(float(x), dtype=torch.float64).clamp_(xs[0], xs[-1])
    hi = torch.bucketize(x_t, xs, right=True).clamp_(1, xs.numel() - 1)
    lo = hi - 1
    span = (xs[hi] - xs[lo]).clamp_(min=1e-9)
    frac = (x_t - xs[lo]) / span
    return float(ys[lo] + frac * (ys[hi] - ys[lo]))


def _sgl_additive_step_time(table, num_requests: int, num_budgets: int):
    floor = table["bias_seconds"] + _sgl_interp_clamped(
        table["bs_probes"], table["alpha_seconds"], float(num_requests)
    )
    m_probes = torch.tensor(table["m_probes"], dtype=torch.float64)
    theta_vals = torch.tensor(table["theta_seconds"], dtype=torch.float64)
    m = (num_requests + torch.arange(num_budgets, dtype=torch.float64)).clamp_(
        min=float(table["m_probes"][0]), max=float(table["m_probes"][-1])
    )
    hi = torch.bucketize(m, m_probes, right=True).clamp_(1, m_probes.numel() - 1)
    lo = hi - 1
    span = (m_probes[hi] - m_probes[lo]).clamp_(min=1e-9)
    frac = (m - m_probes[lo]) / span
    theta_at_m = theta_vals[lo] + frac * (theta_vals[hi] - theta_vals[lo])
    return floor + theta_at_m


def _sgl_compute_verify_token_budget(
    *, history_survival_probs, table, max_verify_len, survival_eps
):
    num_requests = history_survival_probs.shape[0]
    candidates = history_survival_probs[:, :max_verify_len].flatten()
    candidates = candidates[candidates >= survival_eps].to(torch.float64)
    candidates_sorted = torch.sort(candidates, descending=True).values
    prefix_sum = torch.cumsum(candidates_sorted, dim=0)
    tau_star = num_requests + torch.cat([torch.zeros(1, dtype=torch.float64), prefix_sum])
    step_time = _sgl_additive_step_time(table, int(num_requests), int(tau_star.numel()))
    theta = tau_star / step_time
    return int(torch.argmax(theta))


# One certified-table shape shared by both sides. TensorRT-LLM stores ms,
# SGLang stores seconds; the argmax is scale-invariant but the tables are
# built with the honest factor anyway.

M_PROBES = (64, 96, 128, 192, 384, 512, 768, 1536)
THETA_MS = (9.68, 11.08, 11.95, 13.95, 22.09, 23.85, 35.61, 105.64)
BIAS_MS = 25.244443
BS_PROBES = (32, 64, 128, 256)
ALPHA_MS = (0.0, 2.177477, 11.462212, 19.342661)
SGL_BLOCK = 5

TRT_TABLE = SpsCostTable(
    token_counts=M_PROBES,
    step_time_ms=THETA_MS,
    fixed_overhead_ms=BIAS_MS,
    batch_sizes=BS_PROBES,
    batch_overhead_ms=ALPHA_MS,
)
SGL_TABLE = {
    "m_probes": M_PROBES,
    "theta_seconds": tuple(v / 1e3 for v in THETA_MS),
    "bias_seconds": BIAS_MS / 1e3,
    "bs_probes": BS_PROBES,
    "alpha_seconds": tuple(v / 1e3 for v in ALPHA_MS),
}


def _inversion_free_survival(rng, bs):
    """Survivals where every position-0 beats every deeper candidate; the
    shift-exact parity theorem holds only under this condition."""
    conf0 = rng.uniform(0.85, 0.99, size=(bs, 1))
    deeper = rng.uniform(0.25, 0.80, size=(bs, SGL_BLOCK - 1))
    return np.cumprod(np.concatenate([conf0, deeper], axis=1), axis=1)


@pytest.mark.parametrize("bs", [1, 3, 8, 64, 128, 252, 256])
def test_budget_argmax_matches_sglang_up_to_the_floor_shift(bs):
    """Same survival, same table, eps off, no inversions: shift-exact parity."""
    rng = np.random.default_rng(17 + bs)
    for _ in range(5):
        surv = _inversion_free_survival(rng, bs)
        sgl_n = _sgl_compute_verify_token_budget(
            history_survival_probs=torch.tensor(surv, dtype=torch.float32),
            table=SGL_TABLE,
            max_verify_len=SGL_BLOCK,
            survival_eps=0.0,
        )
        trt_m = compute_verify_token_budget(
            survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE, min_verify_len=1
        )
        if sgl_n >= bs:
            assert trt_m == sgl_n - bs, (
                f"bs={bs}: SGLang admits {sgl_n} candidates "
                f"(= floor {bs} + {sgl_n - bs}), TRT-LLM budget {trt_m}"
            )
        else:
            # SGLang went below the one-draft floor; the closest budget
            # TensorRT-LLM can express under min_verify_len=1 is zero.
            assert trt_m == 0


def test_live_pooled_survivals_park_on_the_breakpoint_together():
    """The campaign's operating point: both planners stop at exactly M = 768,
    and tier alignment then rounds down to the capturable rung-2 budget."""
    bs = 252
    surv = np.tile([0.771, 0.693, 0.502, 0.360, 0.315], (bs, 1))
    sgl_n = _sgl_compute_verify_token_budget(
        history_survival_probs=torch.tensor(surv, dtype=torch.float32),
        table=SGL_TABLE,
        max_verify_len=SGL_BLOCK,
        survival_eps=0.0,
    )
    trt_m = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE, min_verify_len=1
    )
    assert trt_m == 768 - 2 * bs
    assert sgl_n - bs == trt_m
    tiered = compute_verify_token_budget(
        survival=surv,
        num_gen_requests=bs,
        cost_table=TRT_TABLE,
        min_verify_len=1,
        allowed_lens=[1, 2, 5],
    )
    assert tiered == bs * 1  # rung-2: one scheduled position past the floor


def test_starving_weak_rows_is_a_known_gap():
    """SGLang can verify NOTHING for a hopeless request while TRT-LLM's
    min_verify_len=1 floor cannot, so SGLang's achievable Theta is better."""
    # Half the batch is strong all the way down, half is dead on arrival.
    # The batch must be big enough that its token range sits on a genuinely
    # rising part of the theta curve, or every candidate is free and the
    # comparison proves nothing.
    bs = 64
    strong = np.tile([0.99, 0.98, 0.97, 0.96, 0.95], (bs // 2, 1))
    dead = np.tile([0.01, 0.008, 0.006, 0.004, 0.002], (bs // 2, 1))
    surv = np.concatenate([strong, dead], axis=0)
    sgl_n = _sgl_compute_verify_token_budget(
        history_survival_probs=torch.tensor(surv, dtype=torch.float32),
        table=SGL_TABLE,
        max_verify_len=SGL_BLOCK,
        survival_eps=0.0,
    )
    trt_m = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE, min_verify_len=1
    )

    def theta_at(tau, tokens):
        return tau / float(TRT_TABLE.step_times(np.asarray([tokens]), bs)[0])

    cand = np.sort(surv.reshape(-1))[::-1]
    sgl_theta = theta_at(bs + cand[:sgl_n].sum(), bs + sgl_n)
    deeper = np.sort(surv[:, 1:].reshape(-1))[::-1]
    trt_theta = theta_at(bs + surv[:, 0].sum() + deeper[:trt_m].sum(), 2 * bs + trt_m)
    assert sgl_theta > trt_theta


def test_eps_filtering_is_a_known_divergence():
    """SGLang drops sub-eps candidates inside the budget; TRT-LLM applies eps
    at allocation time, so its budget can count a candidate the allocator refuses."""
    bs = 16
    surv = np.tile([0.9, 0.4, 0.008, 0.004, 0.002], (bs, 1))
    sgl_n = _sgl_compute_verify_token_budget(
        history_survival_probs=torch.tensor(surv, dtype=torch.float32),
        table=SGL_TABLE,
        max_verify_len=SGL_BLOCK,
        survival_eps=0.01,
    )
    trt_m = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE, min_verify_len=1
    )
    assert sgl_n - bs <= trt_m


# --------------------------------------------------------------------------
# tier-aligned budgets: score with the tau the step actually collects
# --------------------------------------------------------------------------


def _table() -> SpsCostTable:
    """A staircase with genuine risers, so trimming is worth something."""
    token_counts = tuple(range(0, 400, 16))
    step_time_ms = tuple(4.0 + 0.6 * (tok // 96) + 0.004 * tok for tok in token_counts)
    return SpsCostTable(token_counts=token_counts, step_time_ms=step_time_ms, fixed_overhead_ms=1.0)


@pytest.mark.parametrize("tiers", [[1, 2, 5], [1, 3, 5], [1, 5]])
def test_restricted_answer_is_always_realisable(tiers):
    """Every returned budget corresponds to a rung the executor captured."""
    rng = np.random.default_rng(20260803)
    table = _table()
    for _ in range(200):
        bs = int(rng.integers(2, 17))
        survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1]
        n = compute_verify_token_budget(
            survival=survival,
            num_gen_requests=bs,
            cost_table=table,
            min_verify_len=1,
            allowed_lens=tiers,
        )
        assert n % bs == 0, f"budget {n} for {bs} requests is not n*(t-min) for any tier"
        assert (n // bs) + 1 in tiers, f"budget {n} implies tier {(n // bs) + 1}, not in {tiers}"


def test_restricted_never_scores_worse_than_the_uniform_choice():
    """Same grid, better numerator: the chosen rung's realised theta wins;
    at bs=1 the two scorers must pick the same rung outright."""
    rng = np.random.default_rng(31337)
    table = _table()
    tiers = [1, 2, 5]
    for _ in range(300):
        bs = int(rng.integers(1, 17))
        survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1]

        def realised_theta(tier: int) -> float:
            budget = bs * (tier - 1)
            cand = np.sort(survival[:, 1:].reshape(-1))[::-1]
            tau = float(bs) + float(survival[:, :1].sum()) + float(cand[:budget].sum())
            tokens = np.array([bs * (tier + 1)])
            return tau / float(table.step_times(tokens, bs)[0])

        n = compute_verify_token_budget(
            survival=survival,
            num_gen_requests=bs,
            cost_table=table,
            min_verify_len=1,
            allowed_lens=tiers,
        )
        chosen = (n // bs) + 1
        uniform = budget_argmax_over_uniform_lens(
            survival=survival,
            num_gen_requests=bs,
            cost_table=table,
            allowed_lens=tiers,
            min_verify_len=1,
        )
        assert realised_theta(chosen) >= realised_theta(uniform) - 1e-12, (
            f"restricted picked tier {chosen} whose realised theta is below "
            f"the uniform scorer's tier {uniform}"
        )
        if bs == 1:
            assert chosen == uniform, (
                "at bs=1 the uniform and top-k allocations are the same set; "
                "the two scorers must pick the same rung"
            )


# --------------------------------------------------------------------------
# runtime verify-length pin: queue, agree, adopt
# --------------------------------------------------------------------------


def _pin_planner(tiers=(1, 2, 5)):
    table = SpsCostTable(
        token_counts=(0, 512, 768, 1536),
        step_time_ms=(5.0, 68.4, 80.2, 150.5),
        fixed_overhead_ms=1.0,
    )
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1)
    return DSparkVerifyPlanner(
        cfg=cfg,
        cost_table=table,
        tiers=list(tiers),
        detailed_stats=True,
    )


def test_adopting_the_agreed_value_applies_it():
    """Queuing must not change this rank's behaviour before the group agrees;
    adopting the agreed value applies it and consumes the pending request."""
    planner = _pin_planner()
    assert planner.request_verify_len_pin(2) == 2
    # Queued, not yet in force.
    assert planner._forced_verify_len is None
    assert planner.pending_verify_len_pin() == 2
    planner.adopt_verify_len_pin(2)
    assert planner._forced_verify_len == 2
    # Consumed: the next step must not re-broadcast a pin nobody asked for.
    assert planner.pending_verify_len_pin() == -1


def test_a_rank_that_queued_nothing_still_adopts_the_group_value():
    """The endpoint lands on ONE rank; the others learn it from the payload."""
    peer = _pin_planner()
    assert peer.pending_verify_len_pin() == -1
    peer.adopt_verify_len_pin(5)
    assert peer._forced_verify_len == 5


def test_zero_clears_the_pin():
    """Wire protocol: -1 means nobody asked (leave the pin alone); clearing
    needs its own value, 0, because None cannot travel in an int payload."""
    planner = _pin_planner()
    planner.adopt_verify_len_pin(2)
    planner.adopt_verify_len_pin(-1)
    assert planner._forced_verify_len == 2
    assert planner.request_verify_len_pin(None) is None
    assert planner.pending_verify_len_pin() == 0
    planner.adopt_verify_len_pin(0)
    assert planner._forced_verify_len is None


def test_an_uncaptured_length_is_refused_at_the_call_site():
    """Rejected when requested, not on some later step, and never queued."""
    planner = _pin_planner(tiers=(1, 2, 5))
    with pytest.raises(ValueError, match="captured tier ladder"):
        planner.request_verify_len_pin(3)
    with pytest.raises(ValueError, match="outside"):
        planner.request_verify_len_pin(9)
    assert planner.pending_verify_len_pin() == -1
    assert planner._forced_verify_len is None


def test_pinned_steps_hand_every_request_the_same_window():
    """The whole point: the shape a sweep cell can honestly label."""
    planner = _pin_planner()
    bs = 8
    rng = np.random.default_rng(5)
    survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1].copy()
    planner._gather_rows = lambda **_: torch.tensor(survival, dtype=torch.float32)
    planner.adopt_verify_len_pin(2)
    lens = planner.decide_verify_lens(num_gen_requests=bs, reduce_across_ranks=False)
    assert lens == [2] * bs
    assert planner.stats["forced_steps"] == 1


# --------------------------------------------------------------------------
# snapshot instruments: stamp-lag histogram and neutral-row count
# --------------------------------------------------------------------------


def test_detailed_snapshot_instruments_are_opt_in(monkeypatch):
    monkeypatch.delenv("TLLM_DSPARK_DETAILED_STATS", raising=False)
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1)
    planner = DSparkVerifyPlanner(cfg=cfg, tiers=[1, 2, 5])
    planner._note_snapshot_stats(torch.zeros((4, 5)), rows=None)
    assert "snap_rows" not in planner.stats
    assert "snap_neutral_rows" not in planner.stats


def test_neutral_rows_are_counted_and_real_rows_are_not():
    planner = _pin_planner()
    selected = torch.zeros((8, 5))
    selected[3] = 30.0  # the neutral fill: sigmoid ~ 1.0 at any temperature
    planner._note_snapshot_stats(selected, rows=None)
    assert planner.stats["snap_rows"] == 8
    assert planner.stats["snap_neutral_rows"] == 1


def test_lag_histogram_reads_staged_seq_minus_stamp():
    planner = _pin_planner()
    planner._host_stamps = torch.tensor([7, 6, 5, 0], dtype=torch.int32)
    planner._staged_seq = 7
    selected = torch.zeros((4, 5))
    planner._note_snapshot_stats(selected, rows=[0, 1, 2, 3])
    hist = planner.stats["stamp_lag_hist"]
    # Stamp 0 is "never drafted": lag 7 in truth, and it must land visibly at
    # the histogram's far end rather than vanish -- these are exactly the rows
    # the neutral counter should agree about.
    assert hist == {0: 1, 1: 1, 2: 1, 7: 1}


def test_large_and_negative_lags_clamp_instead_of_exploding_the_dict():
    planner = _pin_planner()
    planner._host_stamps = torch.tensor([0, 500], dtype=torch.int32)
    planner._staged_seq = 400
    planner._note_snapshot_stats(torch.zeros((2, 5)), rows=[0, 1])
    assert planner.stats["stamp_lag_hist"] == {8: 1, -2: 1}


def test_counts_accumulate_across_steps():
    planner = _pin_planner()
    planner._host_stamps = torch.tensor([3, 3], dtype=torch.int32)
    planner._staged_seq = 4
    for _ in range(3):
        planner._note_snapshot_stats(torch.full((2, 5), 30.0), rows=[0, 1])
    assert planner.stats["snap_rows"] == 6
    assert planner.stats["snap_neutral_rows"] == 6
    assert planner.stats["stamp_lag_hist"] == {1: 6}


def test_instrument_failure_is_counted_not_raised():
    planner = _pin_planner()
    planner._host_stamps = torch.tensor([1], dtype=torch.int32)
    planner._staged_seq = 1
    # Rows that cannot index the stamp buffer must clamp, and anything worse
    # must be swallowed into the error counter: instruments never kill steps.
    planner._note_snapshot_stats(torch.zeros((1, 5)), rows=[10_000])
    assert planner.stats.get("snap_stats_errors", 0) == 0
    planner._note_snapshot_stats(object(), rows=None)  # type: ignore[arg-type]
    assert planner.stats["snap_stats_errors"] == 1


def test_decide_path_feeds_the_instruments():
    """The counters must fill from the real decide path, not only direct calls."""
    planner = _pin_planner()
    bs = 8
    rng = np.random.default_rng(3)
    survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1].copy()
    planner._gather_rows = lambda **_: torch.tensor(survival, dtype=torch.float32)
    lens = planner.decide_verify_lens(num_gen_requests=bs, reduce_across_ranks=False)
    assert lens is not None
    assert planner.stats["snap_rows"] == bs
    assert planner.stats["snap_neutral_rows"] == 0
    # The local-decision instruments must fill on the same path: one rung per
    # decision, one window per request, and the two must agree -- the windows
    # are the top-k realisation of exactly that rung's budget.
    rung_hist = planner.stats["local_rung_hist"]
    assert sum(rung_hist.values()) == 1
    (rung,) = rung_hist
    assert rung in (1, 2, 5)
    len_hist = planner.stats["local_len_hist"]
    assert sum(len_hist.values()) == bs
    budget = bs * (rung - planner.cfg.min_verify_len)
    extra = sum((k - planner.cfg.min_verify_len) * v for k, v in len_hist.items())
    assert extra <= budget


# --------------------------------------------------------------------------
# reconciliation guards: engine fingerprint and price prediction
# --------------------------------------------------------------------------

LIVE = {"tp": 8, "ep": 8, "attention_dp": True, "block": 5, "max_batch_size": 256}


def _payload(engine):
    return {
        "token_counts": [512, 1536],
        "step_time_ms": [60.0, 150.0],
        "_meta": {"engine": engine} if engine is not None else {},
    }


def test_mismatched_engine_is_refused():
    """Same cell, different max_batch_size: the wrong table must not load."""
    wrong = dict(LIVE, max_batch_size=64)
    with pytest.raises(ValueError, match="different engine configuration"):
        check_table_fingerprint(payload=_payload(wrong), live=dict(LIVE))


def test_fingerprints_that_must_load():
    """Every non-contradicted fingerprint loads; only a contradicted fact
    refuses (case-insensitive strings, unverifiable keys logged, no-meta warns)."""
    check_table_fingerprint(payload=_payload(dict(LIVE)), live=dict(LIVE))
    check_table_fingerprint(
        payload=_payload(dict(LIVE, moe_backend="megamoe_cutedsl")),
        live=dict(LIVE, moe_backend="MEGAMOE_CUTEDSL"),
    )
    check_table_fingerprint(
        payload=_payload(dict(LIVE, image="faf2c60935", geometry="constant_block")), live=dict(LIVE)
    )
    check_table_fingerprint(payload=_payload(None), live=dict(LIVE))


def test_planner_records_the_price_it_paid():
    """Every budget decision must leave behind the step time it assumed, so a
    gap against measured hostStepTimeMS can contradict a wrong table."""
    table = SpsCostTable(
        token_counts=(0, 512, 768, 1536),
        step_time_ms=(5.0, 68.4, 80.2, 150.5),
        fixed_overhead_ms=1.0,
    )
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1)
    planner = DSparkVerifyPlanner(cfg=cfg, cost_table=table, tiers=[1, 2, 5])

    bs = 8
    rng = np.random.default_rng(11)
    survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1].copy()

    # Only the snapshot is faked; calibration stays the constructor's default
    # sigmoid (setting it to None post-construction would bypass the
    # `apply_calibration or torch.sigmoid` fallback and crash the decide path).
    planner._gather_rows = lambda **_: torch.tensor(survival, dtype=torch.float32)

    lens = planner.decide_verify_lens(num_gen_requests=bs, reduce_across_ranks=False)
    assert lens is not None

    steps = planner.stats.get("predicted_steps", 0)
    total = planner.stats.get("predicted_ms_sum", 0.0)
    assert steps == 1 and total > 0.0

    # The recorded price must be the table's own answer for the tokens the
    # decision implies -- anything else and the reconciliation compares two
    # unrelated numbers.
    budget = sum(lens) - bs * cfg.min_verify_len
    tokens = bs * (cfg.min_verify_len + 1) + budget
    expected = float(table.step_times(np.asarray([tokens]), bs)[0])
    assert planner.last_predicted_step_ms == pytest.approx(expected)
    assert total == pytest.approx(expected)
