# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark per-request allocation and SGLang parity tests."""

import itertools

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import (
    SpsCostTable,
    budget_argmax_over_uniform_lens,
    compute_verify_token_budget,
    derive_verify_len_tiers,
)
from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)

BLOCK = 7


def _cfg(**kwargs) -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=BLOCK, **kwargs)


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
