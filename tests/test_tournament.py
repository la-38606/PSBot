import re

from typer.testing import CliRunner

from psbot.cli import app
from psbot.evaluation.tournament import TournamentResult

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI style codes so help-text assertions hold with color on or off."""

    return _ANSI.sub("", text)


def test_tournament_help_lists_options() -> None:
    result = runner.invoke(app, ["evaluate", "tournament", "--help"])
    assert result.exit_code == 0
    stdout = _plain(result.stdout)
    for option in ("--a", "--b", "--n"):
        assert option in stdout


def test_tournament_rejects_unknown_baseline() -> None:
    result = runner.invoke(app, ["evaluate", "tournament", "--a", "garbage"])
    assert result.exit_code != 0


def test_result_win_rate_is_derived() -> None:
    result = TournamentResult(
        agent_a="max-base-power",
        agent_b="random",
        games=100,
        wins_a=87,
        ci_low=0.79,
        ci_high=0.92,
    )
    assert result.win_rate_a == 0.87
