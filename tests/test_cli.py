from typer.testing import CliRunner

from psbot.battle.smoke import SmokeBattleResult
from psbot.cli import app

runner = CliRunner()


def test_help_lists_public_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "smoke", "collect", "replays", "train", "evaluate", "ladder"):
        assert command in result.stdout


def test_doctor_passes_without_server_or_ml_extras() -> None:
    result = runner.invoke(app, ["doctor", "--no-server"])
    assert result.exit_code == 0, result.stdout
    assert "PSBot doctor passed" in result.stdout


def test_battle_smoke_reports_result(monkeypatch) -> None:
    async def fake_smoke_battle() -> SmokeBattleResult:
        return SmokeBattleResult(
            battle_id="battle-gen9randombattle-test",
            turns=23,
            winner="PSBotA",
            player_a="PSBotA",
            player_b="PSBotB",
        )

    monkeypatch.setattr("psbot.battle.smoke.run_smoke_battle", fake_smoke_battle)

    result = runner.invoke(app, ["smoke"])

    assert result.exit_code == 0, result.stdout
    assert (
        "Battle battle-gen9randombattle-test finished in 23 turns; "
        "winner: PSBotA (PSBotA vs PSBotB)"
    ) in result.stdout
