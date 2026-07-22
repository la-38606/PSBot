from typer.testing import CliRunner

from psbot.cli import app

runner = CliRunner()


def test_help_lists_public_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "collect", "replays", "train", "evaluate", "ladder"):
        assert command in result.stdout


def test_doctor_passes_without_server_or_ml_extras() -> None:
    result = runner.invoke(app, ["doctor", "--no-server"])
    assert result.exit_code == 0, result.stdout
    assert "PSBot doctor passed" in result.stdout
