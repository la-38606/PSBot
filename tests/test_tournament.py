import json
import re
from pathlib import Path

from typer.testing import CliRunner

from psbot.agents.baselines import BaselineName
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


def test_tournament_reports_and_saves_result(monkeypatch, tmp_path: Path) -> None:
    async def fake_tournament(
        _a: BaselineName,
        _b: BaselineName,
        _n: int,
    ) -> TournamentResult:
        return TournamentResult(
            agent_a="max-base-power",
            agent_b="random",
            games=10,
            wins_a=8,
            ci_low=0.49,
            ci_high=0.94,
        )

    monkeypatch.setattr(
        "psbot.evaluation.tournament.run_tournament",
        fake_tournament,
    )
    output = tmp_path / "records" / "tournament.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "tournament",
            "--a",
            "max-base-power",
            "--b",
            "random",
            "--n",
            "10",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "max-base-power vs random: 8/10" in result.stdout
    assert f"Saved result to {output}" in result.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["win_rate_a"] == 0.8
