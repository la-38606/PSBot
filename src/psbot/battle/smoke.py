"""One-battle smoke test for the local server."""

from dataclasses import dataclass
from uuid import uuid4

from poke_env import AccountConfiguration, RandomPlayer

from psbot.battle.server import build_server_configuration
from psbot.constants import BATTLE_FORMAT


@dataclass(frozen=True, slots=True)
class SmokeBattleResult:
    """Summary of one completed smoke battle."""

    battle_id: str
    turns: int
    winner: str | None
    player_a: str
    player_b: str


async def run_smoke_battle() -> SmokeBattleResult:
    """Run one random-vs-random battle on the local server."""

    server = build_server_configuration()
    suffix = uuid4().hex[:6]

    player_a = RandomPlayer(
        account_configuration=AccountConfiguration(f"PSBotA{suffix}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
    )

    player_b = RandomPlayer(
        account_configuration=AccountConfiguration(f"PSBotB{suffix}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
    )

    await player_a.battle_against(player_b, n_battles=1)

    if len(player_a.battles) != 1:
        raise RuntimeError(
            f"Expected one completed battle, found {len(player_a.battles)}"
        )

    battle = next(iter(player_a.battles.values()))

    if battle.won:
        winner = player_a.username
    elif battle.lost:
        winner = player_b.username
    else:
        winner = None

    return SmokeBattleResult(
        battle_id=battle.battle_tag,
        turns=battle.turn,
        winner=winner,
        player_a=player_a.username,
        player_b=player_b.username,
    )
