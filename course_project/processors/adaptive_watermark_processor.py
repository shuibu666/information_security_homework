# coding=utf-8
from __future__ import annotations

from statistics import mean
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from watermark_processor import WatermarkLogitsProcessor


class AdaptiveDeltaWatermarkLogitsProcessor(WatermarkLogitsProcessor):
    """Apply a per-step watermark bias based on the entropy of the next-token distribution."""

    def __init__(
        self,
        *args,
        delta_min: float = 0.5,
        delta_max: float = 3.0,
        entropy_floor: float = 0.20,
        delta_exponent: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if delta_min > delta_max:
            raise ValueError("delta_min must be less than or equal to delta_max.")
        if not 0.0 <= entropy_floor <= 1.0:
            raise ValueError("entropy_floor must be between 0.0 and 1.0.")
        if delta_exponent <= 0.0:
            raise ValueError("delta_exponent must be positive.")
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.entropy_floor = entropy_floor
        self.delta_exponent = delta_exponent
        self.delta_history: list[list[float]] = []

    def _compute_step_biases(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        entropies = -(probs * log_probs).sum(dim=-1)
        max_entropy = scores.new_tensor(float(max(self.vocab_size, 2))).log()
        normalized_entropies = (entropies / max_entropy).clamp(0.0, 1.0)
        # The original linear mapping kept most steps too close to delta~1.0 on OPT-2.7B.
        # We lift the uncertainty floor and apply a concave transform so medium-entropy steps
        # receive a meaningfully stronger watermark while still reserving delta_max for the
        # highest-entropy regions.
        boosted_signal = normalized_entropies.clamp(self.entropy_floor, 1.0).pow(self.delta_exponent)
        return self.delta_min + (self.delta_max - self.delta_min) * boosted_signal

    def get_delta_summary(self) -> dict[str, float | None]:
        flattened = [delta for step in self.delta_history for delta in step]
        if not flattened:
            return {
                "avg_step_delta": None,
                "observed_delta_min": None,
                "observed_delta_max": None,
            }
        return {
            "avg_step_delta": mean(flattened),
            "observed_delta_min": min(flattened),
            "observed_delta_max": max(flattened),
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
        for batch_index in range(input_ids.shape[0]):
            batched_greenlist_ids[batch_index] = self._get_greenlist_ids(input_ids[batch_index])

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        step_biases = self._compute_step_biases(scores).to(scores.dtype)
        self.delta_history.append(step_biases.detach().cpu().tolist())

        bias_matrix = green_tokens_mask.to(scores.dtype) * step_biases.unsqueeze(-1)
        return scores + bias_matrix
