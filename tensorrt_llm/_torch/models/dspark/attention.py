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
#
# The DSpark captured-context attention primitives are ported from DeepSeek's
# DeepSpec reference ``inference/kernel.py`` (``sparse_attn``) and
# ``inference/model.py`` (``get_dspark_topk_idxs``); pure-PyTorch, same math.
"""DSpark draft captured-context attention primitives (hardware-agnostic).

Dense sliding-window MLA (``compress_ratio == 0``) for the draft block: queries
come from the block's draft tokens; keys/values are gathered from a rolling
window of projected captured context plus the block's own positions, with an
attention-sink softmax.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F

from ...._utils import is_sm_100f
from ...cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    from ...custom_ops.dspark_attention_custom_op import (
        cute_dsl_dspark_attention,
        is_fused_dspark_attention_supported,
    )
    from ...custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        is_fused_dspark_rmsnorm_rope_supported,
    )

__all__ = [
    "get_dspark_topk_idxs",
    "get_dspark_topk_idxs_batched",
    "dspark_sparse_attn",
    "precompute_dspark_freqs_cis",
    "apply_dspark_rotary",
    "apply_dspark_rotary_batched",
    "dspark_attention_forward",
    "dspark_attention_forward_batched",
]


def precompute_dspark_freqs_cis(
    rope_head_dim: int,
    seqlen: int,
    rope_theta: float = 10000.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Plain (non-YaRN) RoPE table: the dense draft disables YaRN and uses the
    base ``rope_theta``.

    Returns:
        complex64 tensor ``[seqlen, rope_head_dim // 2]``.
    """
    freqs = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device=device) / rope_head_dim)
    )
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_dspark_rotary(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply (or, with ``inverse``, de-apply) rotary embeddings.

    Last dim is adjacent (re, im) pairs; ``inverse`` conjugates. ``x`` is the
    rope-dim slice only: ``[b, s, rd]`` or ``[b, s, h, rd]``, with ``freqs_cis``
    of shape ``[s, rd // 2]``.
    """
    orig_dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        fc = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        fc = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(orig_dtype)


def apply_dspark_rotary_batched(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Per-row variant of :func:`apply_dspark_rotary`: ``freqs_cis`` carries a
    leading batch axis (one set of phases per request). ``x`` is the rope-dim
    slice only: ``[G, s, rd]`` or ``[G, s, h, rd]``, with ``freqs_cis`` of
    shape ``[G, s, rd // 2]``.
    """
    orig_dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    g, s, half = freqs_cis.shape
    if xc.ndim == 3:
        fc = freqs_cis.view(g, s, half)
    else:
        fc = freqs_cis.view(g, s, 1, half)
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(orig_dtype)


@lru_cache(maxsize=64)
def _topk_matrix(window_size: int, block_size: int, start_pos: int) -> torch.Tensor:
    # [min(window, start_pos+1)] context positions in the rolling KV window,
    # followed by [block_size] positions for the current block's own K/V (which
    # the caller appends to the window at offset ``window_size``).
    ctx = torch.arange(min(window_size, start_pos + 1))
    blk = window_size + torch.arange(block_size)
    return torch.cat([ctx, blk]).int()


def get_dspark_topk_idxs(
    window_size: int,
    bsz: int,
    block_size: int,
    start_pos: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-query attended-position indices for the DSpark draft block.

    Every query position attends to the same set: the
    ``min(window_size, start_pos+1)`` most-recent context positions in the
    rolling KV window, then the ``block_size`` block positions stored at offset
    ``window_size``. *Non-causal* within the block, matching the reference.

    Returns:
        int32 tensor ``[bsz, block_size, topk]`` with
        ``topk = min(window_size, start_pos+1) + block_size``.
    """
    assert start_pos > 0, "DSpark draft attention runs at generation (start_pos > 0)"
    matrix = _topk_matrix(int(window_size), int(block_size), int(start_pos)).to(device)
    return matrix.view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def get_dspark_topk_idxs_batched(
    window_size: int,
    block_size: int,
    start_pos: torch.Tensor,
    valid_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sync-free, fixed-shape (CUDA-graph-safe) batched ``get_dspark_topk_idxs``.

    Unlike the scalar :func:`get_dspark_topk_idxs` (whose ``topk`` width
    ``min(window_size, start_pos+1) + block_size`` depends on the host int
    ``start_pos``), this always returns the **fixed** width ``window_size +
    block_size`` and masks the unfilled context slots with ``-1``. The masked
    slots are excluded by :func:`dspark_sparse_attn` exactly as if they were
    absent while the shape remains CUDA-graph safe.

    Every query attends to the actually written circular-window suffix, followed
    by the current-block positions. Without ``valid_len`` this preserves the
    legacy ``start_pos``-only behavior.

    Args:
        window_size: sliding-window length of the captured-context KV cache.
        block_size: number of draft positions per request.
        start_pos: ``[G]`` int tensor of per-request absolute decode positions.
        valid_len: optional ``[G]`` count of actually written rolling-window
            entries. When omitted, preserve the legacy ``start_pos`` mask.

    Returns:
        int32 tensor ``[G, block_size, window_size + block_size]``.
    """
    device = start_pos.device
    g = start_pos.shape[0]
    ctx_cols = torch.arange(window_size, device=device)  # [win]
    if valid_len is None:
        valid = ctx_cols.unsqueeze(0) <= start_pos.unsqueeze(1)  # [G, win]
    else:
        # The valid entries are the contiguous logical suffix ending at
        # start_pos, but their physical slots wrap modulo window_size.
        valid_len = valid_len.clamp(min=0, max=window_size)
        age = torch.remainder(start_pos.unsqueeze(1) - ctx_cols.unsqueeze(0), window_size)
        valid = age < valid_len.unsqueeze(1)
    ctx_idx = torch.where(
        valid, ctx_cols.unsqueeze(0).expand(g, -1), torch.full_like(valid, -1, dtype=torch.long)
    )
    blk_idx = window_size + torch.arange(block_size, device=device)  # [block]
    blk_idx = blk_idx.unsqueeze(0).expand(g, -1)  # [G, block]
    row = torch.cat([ctx_idx, blk_idx], dim=1).to(torch.int32)  # [G, win+block]
    return row.unsqueeze(1).expand(g, block_size, -1).contiguous()


def dspark_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Index-gathered multi-query attention with an attention sink.

    Gathers the ``topk`` KV rows named by ``topk_idxs`` (``-1`` masks a slot),
    softmaxes over them, and adds a per-head learnable *sink* logit that
    participates only in the softmax denominator (an "attend-to-nothing" option
    with a zero value vector). KV is shared across query heads (MQA).

    Args:
        q: ``[b, m, h, d]`` query (``m`` = block_size, ``h`` = heads).
        kv: ``[b, n, d]`` keys/values (shared across heads).
        attn_sink: ``[h]`` per-head sink logits (fp32).
        topk_idxs: ``[b, m, topk]`` int gather indices into ``kv`` (``-1`` masks).
        softmax_scale: scalar applied to the q·k scores (``head_dim ** -0.5``).

    Returns:
        ``[b, m, h, d]`` attention output, in ``q.dtype``.
    """
    b, m, h, d = q.shape
    idx = topk_idxs.long()  # [b, m, topk]
    valid = idx >= 0
    safe = idx.clamp(min=0)

    # Invalid slots read kv[0, :] (via safe.clamp), but masked_fill below
    # zeros their softmax probs, so the einsum nullifies them.
    kv_exp = kv.unsqueeze(1).expand(b, m, kv.shape[1], d)
    gathered = torch.gather(kv_exp, 2, safe.unsqueeze(-1).expand(b, m, safe.shape[-1], d)).float()

    # Scores [b, m, h, topk]; mask invalid slots to -inf before the softmax.
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # Online-softmax max is taken over gathered positions only (the sink is added
    # to the denominator afterwards), matching the kernel's reduce order.
    smax = scores.max(dim=-1, keepdim=True).values  # [b, m, h, 1]
    smax = torch.where(torch.isinf(smax), torch.zeros_like(smax), smax)
    probs = torch.exp(scores - smax)  # masked slots -> exp(-inf) = 0
    sink = torch.exp(attn_sink.to(torch.float32).view(1, 1, h) - smax.squeeze(-1))
    denom = probs.sum(dim=-1) + sink  # [b, m, h]
    out = torch.einsum("bmhk,bmkd->bmhd", probs, gathered) / denom.unsqueeze(-1)
    return out.to(q.dtype)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm matching the DeepSpec reference (fp32 reduce, then * weight)."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


def _rope_last_dims(
    t: torch.Tensor, rope_head_dim: int, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply RoPE to the last ``rope_head_dim`` dims; pass the rest through."""
    nope = t[..., :-rope_head_dim]
    rope = apply_dspark_rotary(t[..., -rope_head_dim:], freqs_cis, inverse=inverse)
    return torch.cat([nope, rope], dim=-1)


def _rope_last_dims_batched(
    t: torch.Tensor, rope_head_dim: int, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Per-row variant of :func:`_rope_last_dims` (``freqs_cis`` has a batch axis)."""
    nope = t[..., :-rope_head_dim]
    rope = apply_dspark_rotary_batched(t[..., -rope_head_dim:], freqs_cis, inverse=inverse)
    return torch.cat([nope, rope], dim=-1)


def _rmsnorm_rope_batched(
    t: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    *,
    num_heads: int = 1,
    apply_weight: bool = True,
    apply_rmsnorm: bool = True,
    inverse_rope: bool = False,
) -> torch.Tensor:
    """Fuse DSpark RMSNorm and last-dimension RoPE when supported."""
    if IS_CUTLASS_DSL_AVAILABLE and is_sm_100f():
        freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
        if is_fused_dspark_rmsnorm_rope_supported(t, weight, freqs_real, num_heads, rope_head_dim):
            return cute_dsl_dspark_rmsnorm_rope(
                t,
                weight,
                freqs_real,
                num_heads,
                rope_head_dim,
                eps,
                apply_weight,
                apply_rmsnorm,
                inverse_rope,
            )

    if apply_rmsnorm:
        if apply_weight:
            t = _rmsnorm(t, weight, eps)
        else:
            t = t * torch.rsqrt(t.square().mean(-1, keepdim=True) + eps)
    elif apply_weight:
        t = (t.float() * weight.float()).to(t.dtype)
    if rope_head_dim > 0:
        t = _rope_last_dims_batched(t, rope_head_dim, freqs_cis, inverse=inverse_rope)
    return t


def dspark_attention_forward(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: int,
    kv_cache: torch.Tensor,
    *,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm_w: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    attn_sink: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
    persist: bool = False,
) -> torch.Tensor:
    """Captured-context DSpark draft attention (generation path, ``start_pos > 0``).

    Port of DeepSpec ``DSparkAttention.forward`` for the dense
    (``compress_ratio == 0``) draft. Weights are plain (already dequantized)
    tensors for ``F.linear``; ``wo_a`` is the raw grouped weight
    ``[n_groups * o_lora_rank, n_heads * head_dim // n_groups]``. ``kv_cache``
    is ``[b, window_size, head_dim]``; the projected ``main_x`` context is
    written at ``start_pos % window_size`` (into a clone unless ``persist``).

    Returns:
        ``[b, block_size, dim]`` attention output.
    """
    assert start_pos > 0, "DSpark draft attention runs at generation (start_pos > 0)"
    b, block, _ = x.shape
    rd = rope_head_dim
    main_freqs = freqs_cis[start_pos : start_pos + 1]
    blk_freqs = freqs_cis[start_pos + 1 : start_pos + 1 + block]

    # Captured-context K/V from main_x (MQA, shared across heads).
    main_kv = _rmsnorm(F.linear(main_x, wkv), kv_norm_w, eps)  # [b, 1, head_dim]
    main_kv = _rope_last_dims(main_kv, rd, main_freqs)

    # Query: low-rank + per-head RMS + RoPE.
    q = _rmsnorm(F.linear(x, wq_a), q_norm_w, eps)
    q = F.linear(q, wq_b).unflatten(-1, (n_heads, head_dim))  # [b, block, h, head_dim]
    # Per-head RMS in the query dtype (matches the reference inline normalization,
    # which is NOT the fp32 RMSNorm path).
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q = _rope_last_dims(q, rd, blk_freqs)

    # Block K/V.
    kv = _rmsnorm(F.linear(x, wkv), kv_norm_w, eps)  # [b, block, head_dim]
    kv = _rope_last_dims(kv, rd, blk_freqs)

    # persist=True writes through to the caller's window (cross-step decode);
    # the default clones so single-shot callers stay pure.
    cache = kv_cache if persist else kv_cache.clone()
    cache[:, start_pos % window_size] = main_kv.squeeze(1)
    kv_full = torch.cat([cache, kv], dim=1)  # [b, window + block, head_dim]
    topk = get_dspark_topk_idxs(window_size, b, block, start_pos, device=x.device)
    o = dspark_sparse_attn(q, kv_full, attn_sink, topk, softmax_scale)  # [b, block, h, head_dim]
    o = _rope_last_dims(o, rd, blk_freqs, inverse=True)

    # Grouped low-rank O projection.
    o = o.reshape(b, block, n_groups, -1)
    wo_a_v = wo_a.view(n_groups, o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a_v)
    return F.linear(o.flatten(2), wo_b)


def dspark_attention_forward_batched(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    valid_len: torch.Tensor | None = None,
    *,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm_w: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    attn_sink: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
    persist: bool = False,
) -> torch.Tensor:
    """Batched, CUDA-graph-safe :func:`dspark_attention_forward`.

    Numerically identical per request to the scalar path, but free of host
    syncs and data-dependent shapes so it can be captured into a CUDA graph
    (the one-engine drafter runs inside the target's graph).

    Args:
        x: ``[G, block, dim]`` block layer input (per gen request).
        main_x: ``[G, 1, hidden]`` projected captured context.
        start_pos: ``[G]`` int tensor of absolute decode positions (> 0).
        kv_cache: ``[N, window_size, head_dim]`` rolling captured-context
            windows, rows indexed by ``slots`` (``persist=True`` writes
            through to the caller's buffer; otherwise a clone is used).
        slots: ``[G]`` int tensor mapping each request to its ``kv_cache`` row.
        valid_len: optional ``[G]`` count of actually written context entries;
            masks holes left when absolute positions are bootstrapped without
            receiving the corresponding DSpark rolling-window state.
        freqs_cis: ``[maxlen, rope_head_dim // 2]`` precomputed plain-RoPE table;
            must satisfy ``maxlen > start_pos.max() + block_size``.

    Returns:
        ``[G, block, dim]`` attention output.
    """
    g, block, _ = x.shape
    if kv_cache.shape[1] != window_size:
        raise ValueError(
            f"kv_cache window extent {kv_cache.shape[1]} does not match window_size {window_size}"
        )
    rd = rope_head_dim
    # Per-request RoPE phases gathered from the fixed table (no host-int slicing).
    main_freqs = freqs_cis[start_pos].unsqueeze(1)  # [G, 1, rd//2]
    blk_pos = start_pos.unsqueeze(1) + 1 + torch.arange(block, device=x.device)  # [G, block]
    blk_freqs = freqs_cis[blk_pos]  # [G, block, rd//2]

    # Captured-context K/V from main_x (MQA, shared across heads).
    main_kv = _rmsnorm_rope_batched(F.linear(main_x, wkv), kv_norm_w, eps, rd, main_freqs)

    # Query: low-rank + per-head RMS + RoPE.
    q = _rmsnorm_rope_batched(F.linear(x, wq_a), q_norm_w, eps, 0, blk_freqs)
    q = F.linear(q, wq_b).unflatten(-1, (n_heads, head_dim))  # [G, block, h, head_dim]
    q = _rmsnorm_rope_batched(
        q,
        kv_norm_w,
        eps,
        rd,
        blk_freqs,
        num_heads=n_heads,
        apply_weight=False,
    )

    # Block K/V.
    kv = _rmsnorm_rope_batched(F.linear(x, wkv), kv_norm_w, eps, rd, blk_freqs)

    # Write the context K/V into the rolling window at slot start_pos%window_size,
    # then attend over [window context | block]. ``persist=True`` writes through to
    # the worker-owned buffer (cross-step decode); otherwise clone so single-shot
    # callers stay pure.
    write_target = kv_cache if persist else kv_cache.clone()
    main_kv_flat = main_kv.squeeze(1).to(write_target.dtype)
    if (
        valid_len is None
        and IS_CUTLASS_DSL_AVAILABLE
        and is_fused_dspark_attention_supported(
            q, main_kv_flat, kv, write_target, slots, start_pos, attn_sink
        )
    ):
        # One custom op performs the rolling-cache write/read, validity handling,
        # QK, attention-sink online softmax, and PV. In particular it creates no
        # topk index, gathered KV, score, or probability tensors.
        o = cute_dsl_dspark_attention(
            q,
            main_kv_flat,
            kv,
            write_target,
            slots,
            start_pos,
            attn_sink,
            softmax_scale,
        )
    else:
        slot_pos = start_pos % window_size  # [G]
        write_target[slots, slot_pos] = main_kv_flat
        cache_rows = write_target[slots]  # [G, window, head_dim]
        kv_full = torch.cat([cache_rows, kv], dim=1)  # [G, window + block, head_dim]
        topk = get_dspark_topk_idxs_batched(window_size, block, start_pos, valid_len)
        o = dspark_sparse_attn(
            q, kv_full, attn_sink, topk, softmax_scale
        )  # [G, block, h, head_dim]
    o = _rmsnorm_rope_batched(
        o,
        kv_norm_w,
        eps,
        rd,
        blk_freqs,
        num_heads=n_heads,
        apply_weight=False,
        apply_rmsnorm=False,
        inverse_rope=True,
    )

    # Grouped low-rank O projection.
    o = o.reshape(g, block, n_groups, -1)
    wo_a_v = wo_a.view(n_groups, o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a_v)
    return F.linear(o.flatten(2), wo_b)
