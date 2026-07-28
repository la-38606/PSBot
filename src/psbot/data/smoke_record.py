from dataclasses import asdict
from pathlib import Path

from psbot.battle.smoke import SmokeBattleResult
from psbot.data.json_record import write_json_record


def save_smoke_result(
    result: SmokeBattleResult,
    destination: Path,
) -> Path:
    """Write a smoke-battle result to a JSON file."""

    return write_json_record(asdict(result), destination)
