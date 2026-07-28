"""JSON persistence for aggregate tournament results."""

from dataclasses import asdict
from pathlib import Path

from psbot.data.json_record import write_json_record
from psbot.evaluation.tournament import TournamentResult


def save_tournament_result(
    result: TournamentResult,
    destination: Path,
) -> Path:
    """Write a tournament result, including its derived win rate, to JSON."""

    result_dict: dict[str, object] = asdict(result)
    result_dict["win_rate_a"] = result.win_rate_a
    return write_json_record(result_dict, destination)
