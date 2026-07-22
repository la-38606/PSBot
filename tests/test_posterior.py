import math

import pytest

from psbot.inference.posterior import summarize_posterior


def test_uniform_posterior_summary() -> None:
    summary = summarize_posterior([1.0, 1.0, 1.0, 1.0])
    assert summary.entropy == pytest.approx(math.log(4))
    assert summary.top_mass == pytest.approx(0.25)
    assert summary.effective_sample_size == pytest.approx(4.0)
    assert summary.plausible_sets == 4


def test_posterior_rejects_zero_mass() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        summarize_posterior([0.0, 0.0])
