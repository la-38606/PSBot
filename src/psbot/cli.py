"""Public PSBot command-line interface."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer

from psbot.config import Settings
from psbot.constants import BATTLE_FORMAT, SHOWDOWN_COMMIT

app = typer.Typer(no_args_is_help=True, help="Build, train, and evaluate PSBot.")
collect_app = typer.Typer(no_args_is_help=True, help="Collect battle trajectories.")
replays_app = typer.Typer(no_args_is_help=True, help="Fetch and reconstruct human replays.")
train_app = typer.Typer(no_args_is_help=True, help="Train classical, BC, and PPO models.")
evaluate_app = typer.Typer(no_args_is_help=True, help="Evaluate agents reproducibly.")
ladder_app = typer.Typer(no_args_is_help=True, help="Run a locked agent on the public ladder.")

app.add_typer(collect_app, name="collect")
app.add_typer(replays_app, name="replays")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(ladder_app, name="ladder")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _planned(feature: str) -> None:
    typer.echo(f"{feature} is scaffolded but not implemented yet.", err=True)
    raise typer.Exit(code=2)


@app.command()
def doctor(
    server: Annotated[
        bool,
        typer.Option("--server/--no-server", help="Check the configured Showdown server."),
    ] = True,
    require_ml: Annotated[
        bool,
        typer.Option(help="Fail if the optional ML dependencies are unavailable."),
    ] = False,
) -> None:
    """Validate the day-one runtime and dependency contract."""

    settings = Settings()
    failures: list[str] = []

    python_ok = sys.version_info[:2] == (3, 11)
    python_state = "ok" if python_ok else "fail"
    typer.echo(
        f"[{python_state}] Python {platform.python_version()} (required: 3.11.x)"
    )
    if not python_ok:
        failures.append("Python 3.11.x is required")

    required = ("poke-env", "numpy", "pydantic", "typer")
    optional_ml = ("torch", "stable-baselines3", "scikit-learn")
    for distribution in required:
        version = _package_version(distribution)
        typer.echo(f"[{'ok' if version else 'fail'}] {distribution}: {version or 'missing'}")
        if version is None:
            failures.append(f"missing dependency: {distribution}")

    for distribution in optional_ml:
        version = _package_version(distribution)
        state = "ok" if version else ("fail" if require_ml else "skip")
        typer.echo(f"[{state}] {distribution}: {version or 'not installed'}")
        if require_ml and version is None:
            failures.append(f"missing ML dependency: {distribution}")

    typer.echo(f"[info] battle format: {BATTLE_FORMAT}")
    typer.echo(f"[info] Showdown commit: {SHOWDOWN_COMMIT}")

    if server:
        try:
            response = httpx.get(
                settings.showdown_health_url,
                timeout=2.0,
            )
            response.raise_for_status()
            typer.echo(f"[ok] Showdown server: {settings.showdown_health_url}")
        except httpx.HTTPError as exc:
            typer.echo(f"[fail] Showdown server: {exc}")
            failures.append("Showdown server is unreachable")

    if failures:
        typer.echo("Doctor found problems:\n- " + "\n- ".join(failures), err=True)
        raise typer.Exit(code=1)

    typer.echo("PSBot doctor passed.")


@collect_app.command("selfplay")
def collect_selfplay(
    config: Annotated[Path, typer.Option()] = Path("configs/data/selfplay.yaml"),
) -> None:
    """Generate local self-play trajectories."""

    _planned(f"Self-play collection with {config}")


@replays_app.command("fetch")
def replay_fetch(
    config: Annotated[Path, typer.Option()] = Path("configs/data/replays.yaml"),
) -> None:
    """Fetch replay metadata and input logs."""

    _planned(f"Replay fetching with {config}")


@replays_app.command("reconstruct")
def replay_reconstruct(
    source: Annotated[Path, typer.Option()] = Path("data/raw/replays"),
) -> None:
    """Reconstruct first-person trajectories from cached replays."""

    _planned(f"Replay reconstruction from {source}")


@train_app.command("classical")
def train_classical(
    config: Annotated[Path, typer.Option()] = Path("configs/bc/classical.yaml"),
) -> None:
    """Train classical action and value baselines."""

    _planned(f"Classical training with {config}")


@train_app.command("bc")
def train_bc(config: Annotated[Path, typer.Option()] = Path("configs/bc/default.yaml")) -> None:
    """Train the behavioral-cloning policy."""

    _planned(f"Behavioral cloning with {config}")


@train_app.command("ppo")
def train_ppo(config: Annotated[Path, typer.Option()] = Path("configs/ppo/default.yaml")) -> None:
    """Fine-tune the policy with PPO self-play."""

    _planned(f"PPO training with {config}")


@evaluate_app.command("tournament")
def evaluate_tournament(
    config: Annotated[Path, typer.Option()] = Path("configs/evaluation/default.yaml"),
) -> None:
    """Run a frozen local tournament."""

    _planned(f"Tournament evaluation with {config}")


@ladder_app.command("run")
def ladder_run(model: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    """Run a locked checkpoint on the public ladder."""

    _planned(f"Public ladder execution with {model}")
