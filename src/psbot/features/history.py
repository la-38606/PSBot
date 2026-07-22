"""Fixed-window, POV-safe battle history."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from psbot.constants import HISTORY_WINDOW


class EventType(StrEnum):
    MOVE = "move"
    SWITCH = "switch"
    DAMAGE = "damage"
    STATUS = "status"
    FAINT = "faint"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BattleEvent:
    turn: int
    actor: str
    event_type: EventType
    value: str | float | None = None


class HistoryBuffer:
    """Maintain the last eight revealed battle events in chronological order."""

    def __init__(self, events: Iterable[BattleEvent] = ()) -> None:
        self._events: deque[BattleEvent] = deque(events, maxlen=HISTORY_WINDOW)

    def append(self, event: BattleEvent) -> None:
        self._events.append(event)

    def snapshot(self) -> tuple[BattleEvent, ...]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)
