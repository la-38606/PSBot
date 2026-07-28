import json
from dataclasses import asdict
from pathlib import Path

from psbot.battle.smoke import SmokeBattleResult


def save_smoke_result(
    result: SmokeBattleResult,
    destination: Path,
) -> Path:
    """Write a smoke-battle result to a JSON file."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    result_dict = asdict(result)

    destination.write_text(
        json.dumps(result_dict, indent=2) + "\n",
        encoding="utf-8",
    )

    return destination
