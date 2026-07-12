"""Exact-token standard KGW scoring used by the final-v1 offline detector."""
from __future__ import annotations

from math import sqrt
from typing import Iterable

import torch

from watermark_processor import WatermarkBase


def standard_kgw_score(prompt_token_ids: Iterable[int], continuation_token_ids: Iterable[int], vocab_size: int, gamma: float, hash_key: int = 15485863) -> dict[str, float | int]:
    prompt = [int(token) for token in prompt_token_ids]
    continuation = [int(token) for token in continuation_token_ids]
    if not prompt or not continuation:
        raise ValueError("standard KGW scoring requires non-empty prompt and continuation token IDs")
    base = WatermarkBase(vocab=list(range(vocab_size)), gamma=gamma, hash_key=hash_key, seeding_scheme="simple_1")
    base.rng = torch.Generator(device="cpu")
    all_ids = prompt + continuation
    green_count = 0
    for position in range(len(prompt), len(all_ids)):
        greenlist = base._get_greenlist_ids(torch.tensor(all_ids[:position], dtype=torch.long))
        green_count += int(all_ids[position] in set(greenlist.tolist()))
    total = len(continuation)
    z_score = (green_count - gamma * total) / sqrt(total * gamma * (1.0 - gamma))
    return {"score": z_score, "z_score": z_score, "num_tokens_scored": total, "num_green_tokens": green_count, "green_fraction": green_count / total}
