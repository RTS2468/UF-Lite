"""Distribution distance metrics for the Fairness gate (TV/KL bound)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations

Distribution = Mapping[str, float]


def _normalize(dist: Distribution) -> dict[str, float]:
    total = float(sum(dist.values()))
    if total <= 0 or any(v < 0 for v in dist.values()):
        raise ValueError("distribution values must be non-negative with a positive sum")
    return {k: v / total for k, v in dist.items()}


def total_variation(p: Distribution, q: Distribution) -> float:
    """Total variation distance in [0, 1]."""
    p, q = _normalize(p), _normalize(q)
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def kl_divergence(p: Distribution, q: Distribution, epsilon: float = 1e-9) -> float:
    """KL(p || q) in nats, smoothed with ``epsilon`` to avoid infinities."""
    p, q = _normalize(p), _normalize(q)
    keys = set(p) | set(q)
    return sum(
        p.get(k, 0.0) * math.log((p.get(k, 0.0) + epsilon) / (q.get(k, 0.0) + epsilon))
        for k in keys
        if p.get(k, 0.0) > 0
    )


def max_pairwise_distance(groups: Mapping[str, Distribution], metric: str = "tv") -> float:
    """Largest distance between any two groups' outcome distributions.

    ``metric`` is ``"tv"`` or ``"kl"`` (symmetrised KL).
    """
    if len(groups) < 2:
        return 0.0
    worst = 0.0
    for a, b in combinations(groups.values(), 2):
        if metric == "kl":
            d = max(kl_divergence(a, b), kl_divergence(b, a))
        else:
            d = total_variation(a, b)
        worst = max(worst, d)
    return worst
