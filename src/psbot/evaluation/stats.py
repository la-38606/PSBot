"""Small, dependency-free evaluation statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a Bernoulli win rate."""

    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("games must be positive and wins must be between zero and games")
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(frozen=True, slots=True)
class PairedSummary:
    pairs: int
    candidate_wins: int
    split_pairs: int
    split_rate: float
    variance_reduction_factor: float


def paired_summary(pair_scores: Iterable[int]) -> PairedSummary:
    """Summarize seat-swapped pair scores, where each score is 0, 1, or 2 wins."""

    scores = tuple(pair_scores)
    if not scores or any(score not in {0, 1, 2} for score in scores):
        raise ValueError("pair scores must be a non-empty sequence containing only 0, 1, or 2")
    split_pairs = sum(score == 1 for score in scores)
    split_rate = split_pairs / len(scores)
    denominator = 2 * (1 - split_rate)
    variance_reduction = math.inf if denominator == 0 else 1 / denominator
    return PairedSummary(
        pairs=len(scores),
        candidate_wins=sum(scores),
        split_pairs=split_pairs,
        split_rate=split_rate,
        variance_reduction_factor=variance_reduction,
    )
