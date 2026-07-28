import json
from pathlib import Path

from psbot.battle.smoke import SmokeBattleResult
from psbot.data.smoke_record import save_smoke_result


def test_save_smoke_result_round_trips(tmp_path: Path) -> None:
    result = SmokeBattleResult(
        battle_id="battle-test",
        turns=42,
        winner="PSBotA",
        player_a="PSBotA",
        player_b="PSBotB",
    )
    destination = tmp_path / "nested" / "smoke.json"

    saved_path = save_smoke_result(result, destination)
    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == destination
    assert data == {
        "battle_id": "battle-test",
        "turns": 42,
        "winner": "PSBotA",
        "player_a": "PSBotA",
        "player_b": "PSBotB",
    }
