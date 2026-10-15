# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output slices preserve loaded weights and FP8 scale-block boundaries."""

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
@pytest.mark.parametrize("backend", ["bf16", "fp8_cute", "fp8_deepgemm"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("start,end", [(0, 576), (576, 1152), (513, 1107)])
@torch.inference_mode()
def test_linear_output_slice(backend, bias, start, end):
    torch.manual_seed(42)
    in_features, out_features = 512, 1152
    quantized = backend != "bf16"
    with torch.device("cuda"):
        linear = Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=torch.bfloat16,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
            if quantized
            else QuantConfig(),
            use_cute_dsl_blockscaling_mm=backend == "fp8_cute",
        )
    weight = torch.randn(out_features, in_features, device="cuda") * 0.02
    checkpoint = {}
    if quantized:
        blocks = weight.view(out_features // 128, 128, in_features // 128, 128)
        scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        checkpoint["weight"] = (weight / expanded).to(torch.float8_e4m3fn)
        checkpoint["weight_scale"] = scale
    else:
        checkpoint["weight"] = weight.bfloat16()
    if bias:
        checkpoint["bias"] = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")
    linear.load_weights([checkpoint])
    linear.post_load_weights()
    original_weight = linear.weight.clone()
    original_scale = linear.weight_scale.clone() if quantized else None
    inputs = torch.randn(31, in_features, dtype=torch.bfloat16, device="cuda")
    expected = linear(inputs)[:, start:end]
    actual = linear.forward_slice(inputs, start, end)
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
    torch.testing.assert_close(
        linear.weight.view(torch.uint8), original_weight.view(torch.uint8), atol=0, rtol=0
    )
    if quantized:
        torch.testing.assert_close(linear.weight_scale, original_scale, atol=0, rtol=0)
