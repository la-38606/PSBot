#Evaluation of N battles between two baseline agents#

from dataclasses import dataclass
from uuid import uuid4

from poke_env import AccountConfiguration

from psbot.agents.baselines import BaselineName, create_baseline
from psbot.battle.server import build_server_configuration
from psbot.evaluation.stats import wilson_interval


@dataclass(frozen=True, slots=True)
class TournamentResult:
    """Outcome of one head-to-head run, reported from agent A's side."""

    agent_a: str
    agent_b: str
    games: int
    wins_a: int
    ci_low: float
    ci_high: float

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games


async def run_tournament(
    name_a: BaselineName, name_b: BaselineName, n_battles: int
) -> TournamentResult:
    """Run n battles between two baselines on the local server."""

    if n_battles < 1:
        raise ValueError("n_battles must be at least 1")

    server = build_server_configuration()
    suffix = uuid4().hex[:6]

    player_a = create_baseline(
        name_a,
        account_configuration=AccountConfiguration(f"PSBotA{suffix}", None),
        server_configuration=server,
    )
    player_b = create_baseline(
        name_b,
        account_configuration=AccountConfiguration(f"PSBotB{suffix}", None),
        server_configuration=server,
    )

    await player_a.battle_against(player_b, n_battles=n_battles)

    games = player_a.n_finished_battles
    if games != n_battles:
        raise RuntimeError(f"expected {n_battles} finished battles, found {games}")

    wins_a = player_a.n_won_battles
    ci_low, ci_high = wilson_interval(wins_a, games)
    return TournamentResult(
        agent_a=name_a.value,
        agent_b=name_b.value,
        games=games,
        wins_a=wins_a,
        ci_low=ci_low,
        ci_high=ci_high,
    )
