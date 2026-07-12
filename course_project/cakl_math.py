"""Numerically small CA-KL mathematics independent of generation frameworks."""
from __future__ import annotations

from math import exp, log


def green_mass_after_bias(p_green_mass: float, delta: float) -> float:
    p = min(max(float(p_green_mass), 0.0), 1.0)
    if delta <= 0.0 or p <= 0.0:
        return p
    multiplier = exp(delta)
    return multiplier * p / (1.0 + (multiplier - 1.0) * p)


def kl_q_to_p(delta: float, p_green_mass: float) -> float:
    """D_KL(Q_delta || P) in nats for a green/red exponential tilt."""
    p = min(max(float(p_green_mass), 0.0), 1.0)
    if delta <= 0.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    normalizer = 1.0 + (exp(delta) - 1.0) * p
    return delta * green_mass_after_bias(p, delta) - log(normalizer)
