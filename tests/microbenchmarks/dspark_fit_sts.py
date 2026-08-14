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
"""Fit per-position sequential temperature scaling for DSpark confidence."""

import argparse
import glob
import json
from pathlib import Path

import torch


def default_temperature_grid() -> torch.Tensor:
    """Return the 41-point log-spaced search grid used by both runtimes."""
    return torch.logspace(-1.0, 1.0, 41, dtype=torch.float64)


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 20,
) -> float:
    """Compute equal-width expected calibration error."""
    probabilities = probabilities.to(torch.float64).flatten()
    labels = labels.to(torch.float64).flatten()
    if probabilities.numel() != labels.numel() or probabilities.numel() == 0:
        raise ValueError("probabilities and labels must have the same non-zero size")
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)
    error = torch.zeros((), dtype=torch.float64)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if index == num_bins - 1 else probabilities < upper
        )
        if torch.any(in_bin):
            weight = in_bin.to(torch.float64).mean()
            error += weight * torch.abs(probabilities[in_bin].mean() - labels[in_bin].mean())
    return float(error)


def fit_sts_temperatures(
    *,
    logits: torch.Tensor,
    prefix_mask: torch.Tensor,
    grid: torch.Tensor,
    min_samples: int,
) -> dict[str, object]:
    """Fit temperatures left-to-right against cumulative prefix survival."""
    logits = logits.to(torch.float64)
    prefix_mask = prefix_mask.to(torch.float64)
    if logits.ndim != 2 or prefix_mask.shape != logits.shape:
        raise ValueError("logits and prefix_mask must have the same [N, K] shape")
    if logits.shape[0] < min_samples:
        raise ValueError(f"STS requires at least {min_samples} rows; got {logits.shape[0]}")
    non_finite_logits = int(torch.count_nonzero(~torch.isfinite(logits)))
    # Keep the conservative 0/1 probabilities used by runtime inference while
    # avoiding NaN losses from BCEWithLogits at literal +/- infinity.
    logits = torch.nan_to_num(
        logits,
        nan=-80.0,
        posinf=80.0,
        neginf=-80.0,
    )

    temperatures: list[float] = []
    ece_before: list[float] = []
    ece_after: list[float] = []
    calibrated_prefix = torch.ones(logits.shape[0], dtype=torch.float64)
    identity_prefix = torch.ones_like(calibrated_prefix)
    for position in range(logits.shape[1]):
        identity_prefix *= torch.sigmoid(logits[:, position])
        labels = prefix_mask[:, position]
        ece_before.append(expected_calibration_error(identity_prefix, labels))

        best_temperature = float(grid[0])
        best_probabilities = calibrated_prefix * torch.sigmoid(
            logits[:, position] / best_temperature
        )
        best_error = expected_calibration_error(best_probabilities, labels)
        for temperature_tensor in grid[1:]:
            temperature = float(temperature_tensor)
            probabilities = calibrated_prefix * torch.sigmoid(logits[:, position] / temperature)
            error = expected_calibration_error(probabilities, labels)
            if error < best_error:
                best_temperature = temperature
                best_probabilities = probabilities
                best_error = error
        temperatures.append(best_temperature)
        ece_after.append(best_error)
        calibrated_prefix = best_probabilities

    return {
        "sts_temperatures": temperatures,
        "temperatures": temperatures,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "num_samples": int(logits.shape[0]),
        "max_draft_len": int(logits.shape[1]),
        "non_finite_logits_sanitized": non_finite_logits,
    }


def load_collected_shards(data_glob: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load opt-in recorder shards and preserve exact logit/label pairing."""
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise ValueError(f"No STS shards matched {data_glob!r}")
    logits, masks = [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("pairing") != "draft_seq_ring":
            raise ValueError(
                f"{path} lacks pairing='draft_seq_ring'; refusing potentially "
                "misaligned confidence/acceptance rows"
            )
        shard_logits = payload.get("logits")
        shard_mask = payload.get("prefix_mask")
        if not isinstance(shard_logits, torch.Tensor) or not isinstance(shard_mask, torch.Tensor):
            raise ValueError(f"{path} must contain tensor logits and prefix_mask")
        if shard_logits.ndim != 2 or shard_mask.shape != shard_logits.shape:
            raise ValueError(f"{path} carries incompatible shard shapes")
        logits.append(shard_logits.to(torch.float32))
        masks.append(shard_mask.to(torch.float32))
    return torch.cat(logits), torch.cat(masks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit DSpark STS temperatures from full-K recorder shards."
    )
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-samples", type=int, default=1000)
    args = parser.parse_args()

    logits, prefix_mask = load_collected_shards(args.data_glob)
    result = fit_sts_temperatures(
        logits=logits,
        prefix_mask=prefix_mask,
        grid=default_temperature_grid(),
        min_samples=args.min_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Fit {result['max_draft_len']} STS temperatures over "
        f"{result['num_samples']} rows -> {output}"
    )


if __name__ == "__main__":
    main()
