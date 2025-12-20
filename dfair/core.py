from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Hashable, Sequence


def _log_safe(p: float) -> float:
    """Small helper so we do not import math just for log."""
    import math

    return math.log(p)


def differential_fairness(
    outcomes: Sequence[Hashable],
    sensitive_attrs: Sequence[Hashable],
    *,
    alpha: float = 1.0,
) -> tuple[float, dict[Hashable, dict[Hashable, float]]]:
    """
    Compute Differential Fairness (Foulds et al., 2019) for a single model output.

    Differential Fairness requires that for every outcome y and for any two
    sensitive groups s, s':

        exp(-epsilon) <= P(Y=y | S=s) / P(Y=y | S=s') <= exp(epsilon)
    """
    if len(outcomes) != len(sensitive_attrs):
        raise ValueError("outcomes and sensitive_attrs must have the same length")
    if len(outcomes) == 0:
        raise ValueError("outcomes and sensitive_attrs cannot be empty")
    if alpha <= 0:
        raise ValueError("alpha must be positive for smoothing")

    unique_outcomes = sorted({o for o in outcomes})
    unique_groups = sorted({s for s in sensitive_attrs})
    num_outcomes = len(unique_outcomes)

    counts = defaultdict(Counter)  # group -> Counter(outcome -> count)
    group_totals = Counter()
    for y, s in zip(outcomes, sensitive_attrs):
        counts[s][y] += 1
        group_totals[s] += 1

    probs: dict[Hashable, dict[Hashable, float]] = {}
    for s in unique_groups:
        total = group_totals[s]
        denom = total + alpha * num_outcomes
        probs[s] = {}
        for y in unique_outcomes:
            probs[s][y] = (counts[s][y] + alpha) / denom

    epsilon = 0.0
    for y in unique_outcomes:
        for g1, g2 in itertools.permutations(unique_groups, 2):
            p1 = probs[g1][y]
            p2 = probs[g2][y]
            log_ratio = abs(_log_safe(p1) - _log_safe(p2))
            epsilon = max(epsilon, log_ratio)

    return epsilon, probs
