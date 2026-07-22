"""Factories for the stable poke-env baseline agents."""

from enum import StrEnum
from typing import Any

from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from psbot.constants import BATTLE_FORMAT


class BaselineName(StrEnum):
    RANDOM = "random"
    MAX_BASE_POWER = "max-base-power"
    SIMPLE_HEURISTIC = "simple-heuristic"


def create_baseline(name: BaselineName, **kwargs: Any) -> Any:
    """Create a baseline with the project battle format applied."""

    if name is BaselineName.RANDOM:
        return RandomPlayer(battle_format=BATTLE_FORMAT, **kwargs)
    if name is BaselineName.MAX_BASE_POWER:
        return MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, **kwargs)
    if name is BaselineName.SIMPLE_HEURISTIC:
        return SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, **kwargs)
    raise AssertionError(f"unhandled baseline: {name}")
