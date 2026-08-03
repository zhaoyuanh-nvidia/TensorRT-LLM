# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused per-head RMSNorm and NeoX RoPE for separate Q and K tensors."""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _separate_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    q_stride_token,
    k_stride_token,
    table_stride_position,
    num_tokens,
    max_positions,
    eps,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    block_tokens: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.program_id(1).to(tl.int64) * block_tokens + tl.arange(0, block_tokens).to(tl.int64)
    row_mask = rows < num_tokens
    offsets = tl.arange(0, half_dim)

    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    if is_k:
        base = k_ptr + rows[:, None] * k_stride_token + local_head * head_dim
        weight_ptr = k_weight_ptr
    else:
        base = q_ptr + rows[:, None] * q_stride_token + local_head * head_dim
        weight_ptr = q_weight_ptr

    x1 = tl.load(base + offsets[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base + half_dim + offsets[None, :], mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )
    sum_squares = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
    inverse_rms = tl.rsqrt(sum_squares / head_dim + eps)

    weight1 = tl.load(weight_ptr + offsets).to(tl.float32)
    weight2 = tl.load(weight_ptr + half_dim + offsets).to(tl.float32)
    y1 = (x1 * inverse_rms[:, None] * weight1[None, :]).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * inverse_rms[:, None] * weight2[None, :]).to(tl.bfloat16).to(tl.float32)

    position = tl.load(positions_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    position = tl.maximum(0, tl.minimum(position, max_positions - 1))
    table_base = position[:, None] * table_stride_position + offsets[None, :]
    cos = (
        tl.load(cos_ptr + table_base, mask=row_mask[:, None], other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    sin = (
        tl.load(sin_ptr + table_base, mask=row_mask[:, None], other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )

    tl.store(
        base + offsets[None, :],
        y1 * cos - y2 * sin,
        mask=row_mask[:, None],
    )
    tl.store(
        base + half_dim + offsets[None, :],
        y2 * cos + y1 * sin,
        mask=row_mask[:, None],
    )


def fused_separate_qk_norm_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen-style per-head RMSNorm and full NeoX RoPE in place.

    ``q`` and ``k`` are independent contiguous ``[tokens, heads * head_dim]``
    BF16 projection outputs. ``cos`` and ``sin`` are the model's FP32
    duplicated-half RoPE tables ``[max_positions, head_dim]``.
    """
    assert q.dim() == 2 and k.dim() == 2
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert q.is_contiguous() and k.is_contiguous()
    assert q.shape[0] == k.shape[0]
    assert q.shape[1] == num_q_heads * head_dim
    assert k.shape[1] == num_kv_heads * head_dim
    assert q_weight.shape == (head_dim,) and k_weight.shape == (head_dim,)
    assert q_weight.device == q.device and k_weight.device == q.device
    assert cos.shape == sin.shape and cos.shape[1] == head_dim
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    assert cos.is_contiguous() and sin.is_contiguous()
    assert positions.numel() == q.shape[0] and positions.device == q.device
    assert head_dim > 0 and head_dim % 2 == 0
    half_dim = head_dim // 2
    assert (half_dim & (half_dim - 1)) == 0

    num_tokens = q.shape[0]
    if num_tokens == 0:
        return q, k

    block_tokens = max(1, min(32, 4096 // head_dim))
    grid = (
        num_q_heads + num_kv_heads,
        triton.cdiv(num_tokens, block_tokens),
    )
    _separate_qk_norm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        positions.reshape(-1),
        q.stride(0),
        k.stride(0),
        cos.stride(0),
        num_tokens,
        cos.shape[0],
        eps,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        half_dim=half_dim,
        block_tokens=block_tokens,
        num_warps=4,
    )
    return q, k
