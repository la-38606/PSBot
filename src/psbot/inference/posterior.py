"""Compact summaries of a posterior over plausible public random sets."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class PosteriorSummary(BaseModel):
    """Features safe to expose to policies after calibration and leakage checks."""

    model_config = ConfigDict(frozen=True)

    entropy: float = Field(ge=0.0)
    top_mass: float = Field(ge=0.0, le=1.0)
    effective_sample_size: float = Field(ge=1.0)
    plausible_sets: int = Field(ge=1)


def summarize_posterior(weights: Sequence[float]) -> PosteriorSummary:
    """Normalize non-negative set weights and compute stable summary statistics."""

    if not weights or any(weight < 0 for weight in weights):
        raise ValueError("posterior weights must be a non-empty, non-negative sequence")
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("posterior weights must contain positive mass")

    probabilities = [weight / total for weight in weights if weight > 0]
    entropy = -math.fsum(probability * math.log(probability) for probability in probabilities)
    squared_mass = math.fsum(probability * probability for probability in probabilities)
    return PosteriorSummary(
        entropy=entropy,
        top_mass=max(probabilities),
        effective_sample_size=1.0 / squared_mass,
        plausible_sets=len(probabilities),
    )
