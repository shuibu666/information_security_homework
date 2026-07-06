# coding=utf-8
from __future__ import annotations

from math import exp, log, sqrt
from statistics import mean
from typing import Iterable

import scipy.stats
import torch

from watermark_processor import WatermarkBase, WatermarkLogitsProcessor


def _mean_or_empty(values: Iterable[float]) -> float | str:
    values = list(values)
    return mean(values) if values else ""


class CAKLWatermarkLogitsProcessor(WatermarkLogitsProcessor):
    """KL-constrained adaptive watermark with optional candidate-aware greenlists."""

    def __init__(
        self,
        *args,
        kl_epsilon: float = 0.02,
        delta_max: float = 3.0,
        candidate_top_p: float = 0.95,
        use_candidate_greenlist: bool = False,
        use_confidence_gate: bool = False,
        entropy_threshold: float = 0.35,
        top1_threshold: float = 0.85,
        binary_search_steps: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if kl_epsilon < 0.0:
            raise ValueError("kl_epsilon must be non-negative.")
        if delta_max < 0.0:
            raise ValueError("delta_max must be non-negative.")
        if not 0.0 < candidate_top_p <= 1.0:
            raise ValueError("candidate_top_p must be in (0, 1].")
        if not 0.0 <= entropy_threshold <= 1.0:
            raise ValueError("entropy_threshold must be between 0 and 1.")
        if not 0.0 <= top1_threshold <= 1.0:
            raise ValueError("top1_threshold must be between 0 and 1.")

        self.kl_epsilon = kl_epsilon
        self.delta_max = delta_max
        self.candidate_top_p = candidate_top_p
        self.use_candidate_greenlist = use_candidate_greenlist
        self.use_confidence_gate = use_confidence_gate
        self.entropy_threshold = entropy_threshold
        self.top1_threshold = top1_threshold
        self.binary_search_steps = binary_search_steps
        self.step_history: list[dict[str, float]] = []

    def _distribution_stats(self, scores: torch.FloatTensor) -> tuple[torch.Tensor, float, float]:
        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs).sum()).detach().cpu())
        max_entropy = log(float(max(self.vocab_size, 2)))
        normalized_entropy = min(max(entropy / max_entropy, 0.0), 1.0)
        top1_prob = float(probs.max().detach().cpu())
        return probs, normalized_entropy, top1_prob

    def _top_p_candidate_ids(self, probs: torch.Tensor) -> torch.LongTensor:
        if self.candidate_top_p >= 1.0:
            return torch.arange(self.vocab_size, device=probs.device)

        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_count = int((cumulative < self.candidate_top_p).sum().item()) + 1
        keep_count = max(1, min(keep_count, sorted_ids.numel()))
        return sorted_ids[:keep_count]

    def _get_candidate_greenlist_ids(self, input_ids: torch.LongTensor, candidate_ids: torch.LongTensor) -> torch.LongTensor:
        if candidate_ids.numel() == 0:
            return candidate_ids

        self._seed_rng(input_ids)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        candidate_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=input_ids.device)
        candidate_mask[candidate_ids] = True
        candidate_permutation = vocab_permutation[candidate_mask[vocab_permutation]]

        greenlist_size = int(candidate_permutation.numel() * self.gamma)
        if candidate_permutation.numel() > 0 and self.gamma > 0:
            greenlist_size = max(1, greenlist_size)
        if self.select_green_tokens:
            return candidate_permutation[:greenlist_size]
        return candidate_permutation[(candidate_permutation.numel() - greenlist_size) :]

    def _kl_for_delta(self, delta: float, p_green_mass: float) -> float:
        if delta <= 0.0 or p_green_mass <= 0.0:
            return 0.0
        p_green_mass = min(max(p_green_mass, 0.0), 1.0)
        normalizer = 1.0 + (exp(delta) - 1.0) * p_green_mass
        q_green_mass = exp(delta) * p_green_mass / normalizer
        return delta * q_green_mass - log(normalizer)

    def _solve_delta(self, p_green_mass: float) -> tuple[float, float]:
        if self.kl_epsilon <= 0.0 or self.delta_max <= 0.0 or p_green_mass <= 0.0:
            return 0.0, 0.0

        max_kl = self._kl_for_delta(self.delta_max, p_green_mass)
        if max_kl <= self.kl_epsilon:
            return self.delta_max, max_kl

        lo, hi = 0.0, self.delta_max
        for _ in range(self.binary_search_steps):
            mid = (lo + hi) / 2.0
            if self._kl_for_delta(mid, p_green_mass) <= self.kl_epsilon:
                lo = mid
            else:
                hi = mid
        return lo, self._kl_for_delta(lo, p_green_mass)

    def _step_greenlist_and_stats(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        probs, normalized_entropy, top1_prob = self._distribution_stats(scores)
        gate_passed = (normalized_entropy >= self.entropy_threshold) and (top1_prob <= self.top1_threshold)
        if not self.use_confidence_gate:
            gate_passed = True

        if self.use_candidate_greenlist:
            candidate_ids = self._top_p_candidate_ids(probs)
            greenlist_ids = self._get_candidate_greenlist_ids(input_ids, candidate_ids)
            candidate_size = int(candidate_ids.numel())
        else:
            greenlist_ids = self._get_greenlist_ids(input_ids)
            candidate_size = self.vocab_size

        if len(greenlist_ids) > 0:
            p_green_mass = float(probs[greenlist_ids].sum().detach().cpu())
        else:
            p_green_mass = 0.0

        if gate_passed:
            delta, kl_value = self._solve_delta(p_green_mass)
        else:
            delta, kl_value = 0.0, 0.0

        stats = {
            "entropy": normalized_entropy,
            "top1_prob": top1_prob,
            "p_green_mass": p_green_mass,
            "delta": delta,
            "kl": kl_value,
            "gate_passed": 1.0 if gate_passed else 0.0,
            "candidate_size": float(candidate_size),
        }
        return greenlist_ids, stats

    def get_generation_summary(self) -> dict[str, float | str]:
        return {
            "avg_kl": _mean_or_empty(row["kl"] for row in self.step_history),
            "avg_delta": _mean_or_empty(row["delta"] for row in self.step_history),
            "avg_entropy": _mean_or_empty(row["entropy"] for row in self.step_history),
            "avg_p_green_mass": _mean_or_empty(row["p_green_mass"] for row in self.step_history),
            "gate_pass_rate": _mean_or_empty(row["gate_passed"] for row in self.step_history),
            "avg_candidate_size": _mean_or_empty(row["candidate_size"] for row in self.step_history),
            "candidate_top_p": self.candidate_top_p if self.use_candidate_greenlist else "",
            "kl_epsilon": self.kl_epsilon,
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        step_deltas = []

        for batch_index in range(input_ids.shape[0]):
            greenlist_ids, stats = self._step_greenlist_and_stats(input_ids[batch_index], scores[batch_index])
            if len(greenlist_ids) > 0 and stats["delta"] > 0.0:
                green_tokens_mask[batch_index][greenlist_ids] = True
            step_deltas.append(stats["delta"])
            self.step_history.append(stats)

        delta_tensor = torch.as_tensor(step_deltas, dtype=scores.dtype, device=scores.device).unsqueeze(-1)
        return scores + green_tokens_mask.to(scores.dtype) * delta_tensor


class CAKLModelAssistedDetector(WatermarkBase):
    """Detector that recomputes prefix distributions for weighted/windowed CA-KL-CG scores."""

    def __init__(
        self,
        *args,
        model,
        tokenizer,
        device: torch.device,
        candidate_top_p: float = 0.95,
        use_candidate_greenlist: bool = False,
        use_confidence_gate: bool = False,
        entropy_threshold: float = 0.35,
        top1_threshold: float = 0.85,
        window_sizes: str = "20,40,80,max",
        z_threshold: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.candidate_top_p = candidate_top_p
        self.use_candidate_greenlist = use_candidate_greenlist
        self.use_confidence_gate = use_confidence_gate
        self.entropy_threshold = entropy_threshold
        self.top1_threshold = top1_threshold
        self.window_sizes = window_sizes
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

    def _distribution_stats(self, scores: torch.FloatTensor) -> tuple[torch.Tensor, float, float]:
        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs).sum()).detach().cpu())
        max_entropy = log(float(max(self.vocab_size, 2)))
        normalized_entropy = min(max(entropy / max_entropy, 0.0), 1.0)
        top1_prob = float(probs.max().detach().cpu())
        return probs, normalized_entropy, top1_prob

    def _top_p_candidate_ids(self, probs: torch.Tensor) -> torch.LongTensor:
        if self.candidate_top_p >= 1.0:
            return torch.arange(self.vocab_size, device=probs.device)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_count = int((cumulative < self.candidate_top_p).sum().item()) + 1
        keep_count = max(1, min(keep_count, sorted_ids.numel()))
        return sorted_ids[:keep_count]

    def _get_candidate_greenlist_ids(self, input_ids: torch.LongTensor, candidate_ids: torch.LongTensor) -> torch.LongTensor:
        if candidate_ids.numel() == 0:
            return candidate_ids

        self._seed_rng(input_ids)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        candidate_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=input_ids.device)
        candidate_mask[candidate_ids] = True
        candidate_permutation = vocab_permutation[candidate_mask[vocab_permutation]]

        greenlist_size = int(candidate_permutation.numel() * self.gamma)
        if candidate_permutation.numel() > 0 and self.gamma > 0:
            greenlist_size = max(1, greenlist_size)
        if self.select_green_tokens:
            return candidate_permutation[:greenlist_size]
        return candidate_permutation[(candidate_permutation.numel() - greenlist_size) :]

    def _score_step(self, prefix_ids: torch.LongTensor, scores: torch.FloatTensor, target_id: int) -> dict[str, float]:
        probs, normalized_entropy, top1_prob = self._distribution_stats(scores)
        gate_passed = (normalized_entropy >= self.entropy_threshold) and (top1_prob <= self.top1_threshold)
        if not self.use_confidence_gate:
            gate_passed = True

        if self.use_candidate_greenlist:
            candidate_ids = self._top_p_candidate_ids(probs)
            greenlist_ids = self._get_candidate_greenlist_ids(prefix_ids, candidate_ids)
            candidate_size = float(candidate_ids.numel())
        else:
            greenlist_ids = self._get_greenlist_ids(prefix_ids)
            candidate_size = float(self.vocab_size)

        p_green_mass = float(probs[greenlist_ids].sum().detach().cpu()) if len(greenlist_ids) > 0 else 0.0
        is_green = 1.0 if len(greenlist_ids) > 0 and bool((greenlist_ids == target_id).any().item()) else 0.0
        weight = normalized_entropy if gate_passed else 0.0
        variance = p_green_mass * (1.0 - p_green_mass)
        return {
            "is_green": is_green,
            "p_green_mass": p_green_mass,
            "weight": weight,
            "variance": variance,
            "entropy": normalized_entropy,
            "gate_passed": 1.0 if gate_passed else 0.0,
            "candidate_size": candidate_size,
        }

    def _compute_weighted_z(self, step_rows: list[dict[str, float]]) -> float:
        numerator = sum(row["weight"] * (row["is_green"] - row["p_green_mass"]) for row in step_rows)
        variance = sum((row["weight"] ** 2) * row["variance"] for row in step_rows)
        if variance <= 0.0:
            return 0.0
        return numerator / sqrt(variance)

    def _parse_window_sizes(self, length: int) -> list[int]:
        sizes = []
        for raw_value in str(self.window_sizes).split(","):
            value = raw_value.strip().lower()
            if not value:
                continue
            if value == "max":
                sizes.append(length)
            else:
                sizes.append(int(value))
        return sorted({size for size in sizes if 0 < size <= length})

    def _compute_winmax_weighted_z(self, step_rows: list[dict[str, float]]) -> float:
        if not step_rows:
            return 0.0

        best_score = 0.0
        for window_size in self._parse_window_sizes(len(step_rows)):
            for start in range(0, len(step_rows) - window_size + 1):
                window_score = self._compute_weighted_z(step_rows[start : start + window_size])
                best_score = max(best_score, window_score)
        return best_score

    def detect(self, prompt_token_ids: torch.LongTensor, generated_token_ids: torch.LongTensor) -> dict[str, float | bool]:
        prompt_token_ids = prompt_token_ids.to(self.device)
        generated_token_ids = generated_token_ids.to(self.device)
        if generated_token_ids.numel() == 0 or prompt_token_ids.numel() == 0:
            return self._empty_result()

        combined_ids = torch.cat([prompt_token_ids, generated_token_ids], dim=0)
        with torch.no_grad():
            logits = self.model(input_ids=combined_ids.unsqueeze(0)).logits[0]

        prompt_len = prompt_token_ids.numel()
        step_rows = []
        for offset, target_id in enumerate(generated_token_ids.tolist()):
            logits_index = prompt_len + offset - 1
            if logits_index < 0:
                continue
            prefix_ids = combined_ids[: prompt_len + offset]
            step_rows.append(self._score_step(prefix_ids, logits[logits_index], int(target_id)))

        if not step_rows:
            return self._empty_result()

        weighted_z = self._compute_weighted_z(step_rows)
        winmax_weighted_z = self._compute_winmax_weighted_z(step_rows)
        p_value = scipy.stats.norm.sf(weighted_z)
        prediction = weighted_z > self.z_threshold
        return {
            "weighted_z_score": weighted_z,
            "winmax_weighted_z_score": winmax_weighted_z,
            "weighted_p_value": p_value,
            "weighted_prediction": prediction,
            "weighted_confidence": (1.0 - p_value) if prediction else 0.0,
            "avg_entropy": _mean_or_empty(row["entropy"] for row in step_rows),
            "avg_p_green_mass": _mean_or_empty(row["p_green_mass"] for row in step_rows),
            "gate_pass_rate": _mean_or_empty(row["gate_passed"] for row in step_rows),
            "avg_candidate_size": _mean_or_empty(row["candidate_size"] for row in step_rows),
        }

    def _empty_result(self) -> dict[str, float | bool]:
        return {
            "weighted_z_score": 0.0,
            "winmax_weighted_z_score": 0.0,
            "weighted_p_value": 1.0,
            "weighted_prediction": False,
            "weighted_confidence": 0.0,
            "avg_entropy": "",
            "avg_p_green_mass": "",
            "gate_pass_rate": "",
            "avg_candidate_size": "",
        }
