"""Exact-token standard KGW scoring used by the final-v1 offline detector."""
from __future__ import annotations

from math import sqrt
from typing import Iterable

import torch

from watermark_processor import WatermarkBase


def standard_kgw_score(
    prompt_token_ids: Iterable[int],
    continuation_token_ids: Iterable[int],
    vocab_size: int,
    gamma: float,
    hash_key: int = 15485863,
    device: str | torch.device = "cpu",
    green_mask_cache: dict[int, torch.BoolTensor] | None = None,
) -> dict[str, float | int]:
    prompt = [int(token) for token in prompt_token_ids]
    continuation = [int(token) for token in continuation_token_ids]
    if not prompt or not continuation:
        raise ValueError("standard KGW scoring requires non-empty prompt and continuation token IDs")
    base = WatermarkBase(vocab=list(range(vocab_size)), gamma=gamma, hash_key=hash_key, seeding_scheme="simple_1")
    # The legacy greenlist uses torch.randperm with a device-local generator.
    # CPU and CUDA generate different permutations for the same seed, so the
    # detector must use the same device class as generation.
    device = torch.device(device)
    base.rng = torch.Generator(device=device)
    all_ids = prompt + continuation
    green_count = 0
    for position in range(len(prompt), len(all_ids)):
        previous_token = all_ids[position - 1]
        green_mask = green_mask_cache.get(previous_token) if green_mask_cache is not None else None
        if green_mask is None:
            # `simple_1` uses only the previous token as its RNG seed.  The
            # cached mask is bit-for-bit equivalent to _get_greenlist_ids but
            # prevents repeating the same 50k-vocabulary permutation.
            base._seed_rng(torch.tensor([previous_token], dtype=torch.long, device=device))
            permutation = torch.randperm(vocab_size, device=device, generator=base.rng)
            green_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            green_mask[permutation[: int(vocab_size * gamma)]] = True
            if green_mask_cache is not None:
                green_mask_cache[previous_token] = green_mask
        green_count += int(green_mask[all_ids[position]].item())
    total = len(continuation)
    z_score = (green_count - gamma * total) / sqrt(total * gamma * (1.0 - gamma))
    return {"score": z_score, "z_score": z_score, "num_tokens_scored": total, "num_green_tokens": green_count, "green_fraction": green_count / total}
