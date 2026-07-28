import json
from pathlib import Path

from psbot.data.tournament_record import save_tournament_result
from psbot.evaluation.tournament import TournamentResult


def test_save_tournament_result_includes_derived_win_rate(tmp_path: Path) -> None:
    result = TournamentResult(
        agent_a="max-base-power",
        agent_b="random",
        games=100,
        wins_a=87,
        ci_low=0.79,
        ci_high=0.92,
    )
    destination = tmp_path / "nested" / "tournament.json"

    saved_path = save_tournament_result(result, destination)
    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == destination
    assert data == {
        "agent_a": "max-base-power",
        "agent_b": "random",
        "games": 100,
        "wins_a": 87,
        "ci_low": 0.79,
        "ci_high": 0.92,
        "win_rate_a": 0.87,
    }
