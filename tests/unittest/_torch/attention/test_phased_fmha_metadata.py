# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention_backend.fmha.phased import _generation_query_metadata


def test_fixed_budget_uses_ragged_lengths_when_blackwell_spec_mode_is_off():
    generation_lengths = torch.tensor([6, 4, 3, 5], dtype=torch.int32)
    position_offsets = torch.arange(24, dtype=torch.int32).view(4, 6)
    metadata = SimpleNamespace(
        confidence_fixed_budget_active=True,
        is_spec_decoding_enabled=False,
        spec_decoding_generation_lengths=generation_lengths,
        spec_decoding_position_offsets_for_cpp=position_offsets,
        max_num_requests=4,
    )

    input_seq_length, actual_lengths, actual_offsets = _generation_query_metadata(
        metadata,
        predicted_tokens_per_seq=6,
        num_gen_tokens=18,
        num_seqs=4,
    )

    assert input_seq_length == 6
    assert actual_lengths is generation_lengths
    assert actual_offsets is position_offsets


def test_regular_generation_preserves_uniform_shape():
    metadata = SimpleNamespace(
        confidence_fixed_budget_active=False,
        is_spec_decoding_enabled=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets_for_cpp=None,
        max_num_requests=4,
    )

    input_seq_length, generation_lengths, position_offsets = _generation_query_metadata(
        metadata,
        predicted_tokens_per_seq=6,
        num_gen_tokens=24,
        num_seqs=4,
    )

    assert input_seq_length == 6
    assert generation_lengths is None
    assert position_offsets is None
