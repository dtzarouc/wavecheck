"""Small statistics helpers for the write-up (no scipy.stats needed beyond the binomial)."""

from __future__ import annotations

import math

from scipy.stats import binom


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    p_hat = k/n; centre = (p_hat + z^2/2n) / (1 + z^2/n);
    half-width = z sqrt(p_hat (1 - p_hat)/n + z^2/4n^2) / (1 + z^2/n).
    Unlike the normal approximation it stays inside [0, 1] and is sensible at p_hat = 0 or 1.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    b = pairs right under condition A and wrong under B; c = the reverse. Under H0 the b + c
    discordant pairs split 50/50, so p = min(1, 2 P[Binom(b + c, 1/2) <= min(b, c)]).
    """
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))
