import pytest

from psbot.evaluation.stats import paired_summary, wilson_interval


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(60, 100)
    assert lower < 0.6 < upper


def test_paired_summary_reports_measured_variance_reduction() -> None:
    summary = paired_summary([0, 1, 1, 2])
    assert summary.pairs == 4
    assert summary.candidate_wins == 4
    assert summary.split_rate == pytest.approx(0.5)
    assert summary.variance_reduction_factor == pytest.approx(1.0)


def test_paired_summary_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="0, 1, or 2"):
        paired_summary([3])
