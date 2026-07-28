"""Base player that separates deciding from plumbing and records reason."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, Player


@dataclass(frozen=True, slots=True)
class Decision:
    order: BattleOrder
    reason: str


@dataclass(frozen=True, slots=True)
class TraceEntry:
    battle_tag: str
    turn: int
    order: str
    reason: str


class PSBotPlayer(Player, ABC):
    """Strategies implement decide(); this class records and never crashes."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._traces: dict[str, list[TraceEntry]] = {}

    @abstractmethod
    def decide(self, battle: AbstractBattle) -> Decision:
        """Pick an order and say why. Runs once per request."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        try:
            decision = self.decide(battle)
        except Exception as exc:  # never forfeit to a strategy bug
            decision = Decision(
                order=self.choose_random_move(battle),
                reason=f"FALLBACK random: decide() raised {type(exc).__name__}: {exc}",
            )
        self._traces.setdefault(battle.battle_tag, []).append(
            TraceEntry(
                battle_tag=battle.battle_tag,
                turn=battle.turn,
                order=str(decision.order),
                reason=decision.reason,
            )
        )
        return decision.order

    def drain_traces(self, battle_tag: str) -> tuple[TraceEntry, ...]:
        """Return and forget one battle's trace.

        The logging pipeline calls this once per finished battle, so trace
        memory is bounded by battles in flight, not battles ever played.
        """

        return tuple(self._traces.pop(battle_tag, []))

    def peek_traces(self, battle_tag: str) -> tuple[TraceEntry, ...]:
        """Return one battle's trace without clearing it (debugging, tests)."""

        return tuple(self._traces.get(battle_tag, []))
