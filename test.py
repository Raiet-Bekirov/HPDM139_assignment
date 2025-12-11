from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Hashable, Iterable, Sequence


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

    Parameters
    ----------
    outcomes:
        Model outputs for each example (e.g., predicted label or score bucket).
    sensitive_attrs:
        Sensitive attribute value for each example (e.g., group A/B, gender).
    alpha:
        Symmetric Dirichlet prior used for Laplace smoothing to avoid zero
        probabilities. alpha=1.0 is the common "add-one" smoothing.

    Returns
    -------
    epsilon:
        The maximal log-probability ratio across all outcomes and group pairs.
    probs:
        Nested dictionary mapping group -> outcome -> P(Y=y | S=group).
    """
    if len(outcomes) != len(sensitive_attrs):
        raise ValueError("outcomes and sensitive_attrs must have the same length")
    if len(outcomes) == 0:
        raise ValueError("outcomes and sensitive_attrs cannot be empty")
    if alpha <= 0:
        raise ValueError("alpha must be positive for smoothing")

    # Collect possible values.
    unique_outcomes = sorted({(o) for o in outcomes})
    unique_groups = sorted({(s) for s in sensitive_attrs})
    num_outcomes = len(unique_outcomes)

    # Count occurrences per group/outcome.
    counts = defaultdict(Counter)  # group -> Counter(outcome -> count)
    group_totals = Counter()
    for y, s in zip(outcomes, sensitive_attrs):
        counts[s][y] += 1
        group_totals[s] += 1

    # Compute smoothed conditional probabilities P(Y=y | S=s).
    probs: dict[Hashable, dict[Hashable, float]] = {}
    for s in unique_groups:
        total = group_totals[s]
        denom = total + alpha * num_outcomes
        probs[s] = {}
        for y in unique_outcomes:
            probs[s][y] = (counts[s][y] + alpha) / denom

    # Max log-ratio across all group pairs and outcomes.
    epsilon = 0.0
    for y in unique_outcomes:
        for g1, g2 in itertools.permutations(unique_groups, 2):
            p1 = probs[g1][y]
            p2 = probs[g2][y]
            # Smoothing guarantees p1 and p2 are positive.
            log_ratio = abs(_log_safe(p1) - _log_safe(p2))
            epsilon = max(epsilon, log_ratio)

    return epsilon, probs


def demo() -> None:
    """
    Minimal example showing how to evaluate Differential Fairness.

    Group A has higher positive rate than group B. Smaller epsilon means closer
    parity; epsilon=0 would mean identical conditional distributions.
    """
    # Fake model outputs and sensitive groups for 10 examples.
    # Y=1 means a positive decision (e.g., approved), Y=0 negative.
    y_pred = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    epsilon, probs = differential_fairness(y_pred, groups, alpha=1.0)

    print("Conditional probabilities P(Y=y | S=s):")
    for group, p in probs.items():
        for outcome, prob in p.items():
            print(f"  Group {group}, Y={outcome}: {prob:.3f}")

    print(f"\nDifferential Fairness epsilon: {epsilon:.4f}")
    print("Lower epsilon means more similar behavior across groups.")


if __name__ == "__main__":
    demo()